from lib.pga import PGA
from lib.surrogates import set_module_standard_backward_, soften_module_inplace_


class LocalGradientAscent:
    def __init__(self, model, alpha=None, steps=1, eps=None):
        self.model = model
        self.atk = PGA(self.model, alpha=alpha, steps=steps, eps=eps)
        self.atk.set_mode_targeted_by_label()

    def attribute(self, inputs, target):
        inputs = inputs.to(self.atk.device)

        adv_inputs = self.atk(inputs, target)

        attributions = adv_inputs - inputs

        return attributions


class LocalRelevanceAscent(LocalGradientAscent):
    # NOTE: This modifies the model IN PLACE, but should not affect forward nor backward passes, as we leave standard_backward=True
    def __init__(self, model, temperatures, alpha=None, steps=1, eps=None):
        super().__init__(model, alpha=alpha, steps=steps, eps=eps)

        self.temperatures = temperatures
        soften_module_inplace_(self.model, temperatures=self.temperatures)
        set_module_standard_backward_(
            self.model, standard_backward=True
        )  # don't change the model behavior!

    def attribute(self, inputs, target):
        set_module_standard_backward_(self.model, standard_backward=False)

        attributions = super().attribute(inputs, target)

        set_module_standard_backward_(self.model, standard_backward=True)

        return attributions
