import numpy as np
import quantus
from quantus.helpers import utils


class InfidelityOptimalScaling(quantus.Infidelity):
    def evaluate_batch(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ):
        """
        This method performs XAI evaluation on a single batch of explanations.
        Infidelity is always computed with Captum-style optimal scaling:
            beta = E[a * b] / E[a^2]
            score = E[(beta * a - b)^2]

        where:
            a = <attribution, input_perturbation>
            b = f(x) - f(x - I)

        Parameters
        ----------
        model: ModelInterface
            A ModelInterface that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        y_batch: np.ndarray
            The output to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        kwargs:
            Unused.

        Returns
        -------
        scores_batch:
            The evaluation results.
        """
        # Prepare shapes. Expand a_batch if not the same shape.
        if x_batch.shape != a_batch.shape:
            a_batch = np.broadcast_to(a_batch, x_batch.shape)

        # Flatten attributions.
        batch_size = a_batch.shape[0]
        a_batch = a_batch.reshape(batch_size, -1)

        # Predict on original input.
        x_input = model.shape_input(
            x_batch, x_batch.shape, channel_first=True, batched=True
        )
        y_pred = model.predict(x_input)[np.arange(batch_size), y_batch]

        # Aggregate Captum polynomial terms across all perturbations.
        # a := attribution dot perturbation
        # b := output difference
        agg_a2 = np.zeros(batch_size, dtype=np.float64)
        agg_ab = np.zeros(batch_size, dtype=np.float64)
        agg_b2 = np.zeros(batch_size, dtype=np.float64)

        total_perturbations = 0

        for _ in range(self.n_perturb_samples):
            for patch_size in self.perturb_patch_sizes:
                x_perturbed = x_batch.copy()
                x_perturbed_h, x_perturbed_w = x_perturbed.shape[-2:]

                padding_h = utils.get_padding_size(x_perturbed_h, patch_size)
                padding_w = utils.get_padding_size(x_perturbed_w, patch_size)

                x_perturbed_pad = utils._pad_array(
                    x_perturbed,
                    ((0, 0), (0, 0), padding_h, padding_w),
                    mode="edge",
                    padded_axes=np.arange(len(x_perturbed.shape)),
                )
                x_perturbed_pad_shape = x_perturbed_pad.shape

                for x_indices in utils.get_block_indices(x_perturbed_pad, patch_size):
                    # Perturb input by block indices of certain patch size.
                    x_perturbed_pad = self.perturb_func(
                        arr=x_perturbed.reshape(batch_size, -1),
                        indices=x_indices,
                    )
                    x_perturbed_pad = x_perturbed_pad.reshape(*x_perturbed_pad_shape)

                    x_perturbed = x_perturbed_pad[
                        :,
                        :,
                        padding_h[0] : x_perturbed_pad.shape[2] - padding_h[1],
                        padding_w[0] : x_perturbed_pad.shape[3] - padding_w[1],
                    ]

                    # Predict on perturbed input.
                    x_input = model.shape_input(
                        x_perturbed, x_batch.shape, channel_first=True, batched=True
                    )
                    y_pred_perturb = model.predict(x_input)[
                        np.arange(batch_size), y_batch
                    ]

                    # b = f(x) - f(x - I)
                    pred_delta = y_pred - y_pred_perturb

                    # a = <attr, I>, where I = x - x_perturbed
                    x_diff = x_batch - x_perturbed
                    a_diff = a_batch * x_diff.reshape(batch_size, -1)
                    a_sum = np.sum(a_diff, axis=-1)

                    # Aggregate terms for normalized infidelity.
                    agg_a2 += a_sum**2
                    agg_ab += a_sum * pred_delta
                    agg_b2 += pred_delta**2

                    total_perturbations += 1

        if total_perturbations == 0:
            return np.zeros(batch_size, dtype=np.float64)

        # Captum-style safe_div:
        # beta = agg_ab / agg_a2, with default denominator 1.0 when agg_a2 == 0
        beta = np.divide(
            agg_ab,
            np.where(agg_a2 != 0, agg_a2, 1.0),
        )

        # (beta*a - b)^2 = beta^2 * a^2 - 2*beta*ab + b^2
        infidelity = beta**2 * agg_a2 - 2.0 * beta * agg_ab + agg_b2
        # infidelity = agg_a2 - 2.0 * agg_ab + agg_b2

        # Convert sums to expectations.
        infidelity = infidelity / total_perturbations

        return infidelity


class FaithfulnessCorrelationPatches(quantus.FaithfulnessCorrelation):
    def __init__(
        self,
        perturb_patch_sizes=(56,),
        # nr_runs=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.perturb_patch_sizes = perturb_patch_sizes
        # self.nr_runs = nr_runs

    def evaluate_batch(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ):
        if x_batch.shape != a_batch.shape:
            a_batch = np.broadcast_to(a_batch, x_batch.shape)

        batch_size = a_batch.shape[0]

        x_input = model.shape_input(
            x_batch, x_batch.shape, channel_first=True, batched=True
        )
        y_pred = model.predict(x_input)[np.arange(batch_size), y_batch]

        pred_deltas = []
        att_sums = []

        for _ in range(self.nr_runs):
            for patch_size in self.perturb_patch_sizes:
                x_h, x_w = x_batch.shape[-2:]

                padding_h = utils.get_padding_size(x_h, patch_size)
                padding_w = utils.get_padding_size(x_w, patch_size)

                x_padded = utils._pad_array(
                    x_batch,
                    ((0, 0), (0, 0), padding_h, padding_w),
                    mode="edge",
                    padded_axes=np.arange(len(x_batch.shape)),
                )
                a_padded = utils._pad_array(
                    a_batch,
                    ((0, 0), (0, 0), padding_h, padding_w),
                    mode="edge",
                    padded_axes=np.arange(len(a_batch.shape)),
                )

                block_indices = list(utils.get_block_indices(x_padded, patch_size))
                if len(block_indices) == 0:
                    continue

                # # different random patch for each example in batch
                # a_ix = np.stack(
                #     [
                #         np.asarray(
                #             block_indices[np.random.randint(len(block_indices))]
                #         ).reshape(-1)
                #         # np.asarray(block_indices[2]).reshape(-1)
                #         for _ in range(batch_size)
                #     ],
                #     axis=0,
                # )
                # same random patch for all examples in batch
                a_ix = block_indices[np.random.randint(len(block_indices))]

                x_padded_shape = x_padded.shape
                x_perturbed_padded = self.perturb_func(
                    arr=x_padded.reshape(batch_size, -1),
                    indices=a_ix,
                ).reshape(*x_padded_shape)

                x_perturbed = x_perturbed_padded[
                    :,
                    :,
                    padding_h[0] : x_perturbed_padded.shape[2] - padding_h[1],
                    padding_w[0] : x_perturbed_padded.shape[3] - padding_w[1],
                ]

                x_input = model.shape_input(
                    x_perturbed, x_batch.shape, channel_first=True, batched=True
                )
                y_pred_perturb = model.predict(x_input)[np.arange(batch_size), y_batch]

                pred_deltas.append(
                    np.asarray(y_pred - y_pred_perturb).reshape(batch_size)
                )

                flat_a_padded = a_padded.reshape(batch_size, -1)
                att_sums.append(
                    flat_a_padded[np.arange(batch_size)[:, None], a_ix].sum(axis=-1)
                )

        if len(pred_deltas) == 0:
            return np.zeros(batch_size, dtype=np.float64).tolist()

        pred_deltas = np.stack(pred_deltas, axis=1)
        att_sums = np.stack(att_sums, axis=1)

        similarity = self.similarity_func(
            a=att_sums,
            b=pred_deltas,
            batched=True,
        )

        return similarity.tolist()
