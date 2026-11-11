import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent.prompt_refiner import PromptRefinementError, refine_request
from agent.subagents.orchestrator import AutonomousPhysicsOrchestrator


class PromptRefinementTests(unittest.TestCase):
    def setUp(self):
        models = patch('agent.prompt_refiner.config.PROMPT_REFINER_MODELS', ['draft-model', 'critic-model'])
        models.start()
        self.addCleanup(models.stop)

    def test_drafts_cross_checks_and_revises_with_distinct_models(self):
        client = Mock()
        chat = client.chats.create.return_value
        chat.send_message.side_effect = [
            Mock(text='Draft', function_calls=None),
            Mock(text='Ask for luminosity. Preserve the no-ML constraint.', function_calls=None),
            Mock(text='Final brief: verify luminosity. No ML.', function_calls=None),
        ]
        with patch('agent.utils._apply_rate_limit_throttle'):
            brief = refine_request(client, 'Extract cross sections. No ML.', 'events.csv', 'Include uncertainties')
        self.assertIn('Extract cross sections. No ML.', brief)
        self.assertIn('Final brief: verify luminosity. No ML.', brief)
        self.assertIn('Independent review: critic-model', brief)
        self.assertEqual(
            [call.kwargs['model'] for call in client.chats.create.call_args_list],
            ['draft-model', 'critic-model', 'draft-model'],
        )
        payloads = [json.loads(call.args[0]) for call in chat.send_message.call_args_list]
        self.assertEqual(payloads[0]['target_criteria'], 'Include uncertainties')
        self.assertEqual(payloads[1]['draft'], 'Draft')
        self.assertEqual(payloads[1]['context']['original_request'], 'Extract cross sections. No ML.')
        self.assertEqual(payloads[2]['independent_critique'], 'Ask for luminosity. Preserve the no-ML constraint.')
        for call in client.chats.create.call_args_list:
            self.assertIsNone(call.kwargs['config'].tools)

    def test_requires_two_distinct_models(self):
        for models in ([], ['only-model'], ['only-model', 'models/only-model']):
            with self.subTest(models=models), patch('agent.prompt_refiner.config.PROMPT_REFINER_MODELS', models):
                client = Mock()
                with self.assertRaisesRegex(PromptRefinementError, 'at least two distinct'):
                    refine_request(client, 'Plot energy')
                client.chats.create.assert_not_called()

    def test_review_failure_stops_without_single_model_fallback(self):
        with patch('agent.prompt_refiner._generate', side_effect=['Draft', RuntimeError('quota')]):
            with self.assertRaisesRegex(PromptRefinementError, 'independent review failed'):
                refine_request(Mock(), 'Plot energy')

    def test_failed_drafter_can_be_replaced_and_independently_reviewed(self):
        with patch('agent.prompt_refiner.config.PROMPT_REFINER_MODELS', ['unavailable', 'second', 'third']), \
             patch('agent.prompt_refiner._generate', side_effect=[RuntimeError('quota'), 'Draft', RuntimeError('quota'), 'Critique', 'Final']) as generate:
            brief = refine_request(Mock(), 'Plot energy')
        self.assertIn('Draft: second', brief)
        self.assertIn('Independent review: third', brief)
        self.assertEqual([call.args[1] for call in generate.call_args_list], ['unavailable', 'second', 'unavailable', 'third', 'second'])

    def test_empty_response_and_tool_calls_are_rejected(self):
        for response in (Mock(text='', function_calls=None), Mock(text='Draft', function_calls=[Mock()])):
            with self.subTest(response=response), patch('agent.prompt_refiner.ResilientChatSession') as session:
                session.return_value.send_message.return_value = response
                with self.assertRaisesRegex(PromptRefinementError, 'draft failed'):
                    refine_request(Mock(), 'Plot energy')

    def test_failed_final_revision_stops_the_request(self):
        with patch('agent.prompt_refiner._generate', side_effect=['Draft', 'Critique', RuntimeError('quota'), RuntimeError('quota')]):
            with self.assertRaisesRegex(PromptRefinementError, 'final revision failed'):
                refine_request(Mock(), 'Plot energy')

    def test_pipeline_passes_brief_to_both_agents_and_archives_it(self):
        for existing_plan in (None, 'Existing methodology'):
            with self.subTest(existing_plan=existing_plan), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                logger = Mock(run_dir=base)
                methodologist = Mock()
                methodologist.plan.return_value = 'New methodology'
                methodologist.peer_review.return_value = {'approved': True, 'review': 'Approved'}
                experimenter = Mock()
                experimenter.execute_plan.return_value = 'Results'
                with patch('agent.subagents.orchestrator.RunLogger', return_value=logger), \
                     patch('agent.subagents.orchestrator.MethodologistAgent', return_value=methodologist), \
                     patch('agent.subagents.orchestrator.ExperimenterAgent', return_value=experimenter), \
                     patch('agent.subagents.orchestrator.refine_request', return_value='Detailed task brief') as refine, \
                     patch('agent.subagents.orchestrator.set_active_run_context'), \
                     patch('agent.subagents.orchestrator.set_theorist_consult_handler'), \
                     patch('agent.subagents.orchestrator.__file__', str(base / 'agent' / 'subagents' / 'orchestrator.py')):
                    events = Mock()
                    pipeline = AutonomousPhysicsOrchestrator(on_crosstalk=events)
                    result = pipeline.run_pipeline('events.csv', 'Fit energy', existing_plan=existing_plan)
                self.assertEqual(result['status'], 'SUCCESS_PEER_REVIEW_APPROVED')
                self.assertEqual((base / 'task_brief.md').read_text(), 'Detailed task brief')
                self.assertEqual(refine.call_args.kwargs['user_request'], 'Fit energy')
                if existing_plan:
                    methodologist.plan.assert_not_called()
                else:
                    self.assertIn('Detailed task brief', methodologist.plan.call_args.kwargs['user_description'])
                self.assertIn('Detailed task brief', experimenter.execute_plan.call_args.kwargs['methodology_plan'])
                self.assertEqual(logger.finalize_run.call_args.kwargs['task'], 'Fit energy')
                events.assert_any_call('Input Refinement', 'Analysis Agents', 'Detailed task brief')

    def test_failed_cross_check_prevents_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            methodologist = Mock()
            experimenter = Mock()
            with patch('agent.subagents.orchestrator.RunLogger', return_value=Mock(run_dir=Path(directory))), \
                 patch('agent.subagents.orchestrator.MethodologistAgent', return_value=methodologist), \
                 patch('agent.subagents.orchestrator.ExperimenterAgent', return_value=experimenter), \
                 patch('agent.subagents.orchestrator.refine_request', side_effect=PromptRefinementError('Review unavailable')), \
                 patch('agent.subagents.orchestrator.set_active_run_context'), \
                 patch('agent.subagents.orchestrator.set_theorist_consult_handler'):
                pipeline = AutonomousPhysicsOrchestrator()
                with self.assertRaisesRegex(PromptRefinementError, 'Review unavailable'):
                    pipeline.run_pipeline('events.csv', 'Fit energy')
            methodologist.plan.assert_not_called()
            experimenter.execute_plan.assert_not_called()

    def test_general_assistant_sends_brief_and_keeps_original_in_memory(self):
        from agent.core import Agent
        agent = Agent()
        client = Mock()
        response = Mock(text='Answer', function_calls=None)
        with patch.object(agent, '_init_client', return_value=client), \
             patch('agent.core.refine_request', return_value='Expanded request') as refine, \
             patch('agent.core.send_message_with_retry', return_value=response) as send:
            self.assertEqual(agent.run('Describe my dataset'), 'Answer')
        refine.assert_called_once_with(client, 'Describe my dataset')
        self.assertEqual(send.call_args.args[1], 'Expanded request')
        self.assertEqual(agent.memory.get_messages()[0]['parts'], ['Describe my dataset'])


if __name__ == '__main__':
    unittest.main()
