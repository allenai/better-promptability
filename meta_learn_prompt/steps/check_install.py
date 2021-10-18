from tango import Step


@Step.register("check_install")
class CheckInstall(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self) -> None:
        import torch

        if torch.cuda.is_available():
            print("All good! CUDA is available :)")
        else:
            print("All good! No CUDA though :/")
