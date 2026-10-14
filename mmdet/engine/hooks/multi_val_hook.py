import torch
from mmengine.evaluator import Evaluator
from mmengine.hooks import Hook
from mmengine.runner import Runner, ValLoop
from mmdet.registry import HOOKS


@HOOKS.register_module()
class MultiValHook(Hook):
    def __init__(self, dataloaders, evaluators):
        self.dataloaders = dataloaders
        self.evaluators = evaluators

    # def before_train_epoch(self, runner) -> None:
    #     model = runner.model
    #     model.eval()
    #
    #     for idx, (dataloader, evaluator) in enumerate(zip(self.dataloaders, self.evaluators)):
    #
    #         # build evaluator if config
    #         if not isinstance(evaluator, Evaluator):
    #             evaluator = runner.build_evaluator(evaluator)
    #
    #         # build dataloader if config
    #         if isinstance(dataloader, dict):
    #             dataloader = runner.build_dataloader(dataloader)
    #
    #         # build a validation loop
    #         val_loop = ValLoop(
    #             runner=runner,
    #             dataloader=dataloader,
    #             evaluator=evaluator
    #         )
    #
    #         # run validation
    #         metrics = val_loop.run()
    #
    #         # log metrics with dataset prefix
    #         for k, v in metrics.items():
    #             runner.message_hub.update_scalar(
    #                 f'val{idx}/{k}', v
    #             )
    #
    #         runner.logger.info(
    #             f'MultiValHook: Validation set {idx} results: {metrics}'
    #         )
    #
    #     model.train()


    def after_train_epoch(self, runner: Runner):
        model = runner.model
        model.eval()

        for idx, (dataloader, evaluator) in enumerate(zip(self.dataloaders, self.evaluators)):

            # build evaluator if config
            if not isinstance(evaluator, Evaluator):
                evaluator = runner.build_evaluator(evaluator)

            # build dataloader if config
            if isinstance(dataloader, dict):
                dataloader = runner.build_dataloader(dataloader)

            # build a validation loop
            val_loop = ValLoop(
                runner=runner,
                dataloader=dataloader,
                evaluator=evaluator
            )

            # run validation
            metrics = val_loop.run()

            # log metrics with dataset prefix
            for k, v in metrics.items():
                runner.message_hub.update_scalar(
                    f'val{idx}/{k}', v
                )

            runner.logger.info(
                f'MultiValHook: Validation set {idx} results: {metrics}'
            )

        model.train()