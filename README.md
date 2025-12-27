# NOTES

- We don't implement the directional faithfulness as Quantus metric - because Quantus enforces the casting to cpu and numpy, which is awkward. Instad, we compute directional faithfulness in Pytorch, using attribution methods straight from Captum (especially since Quantus is a simple wrapper for Captum).
- But we do use Quantus adapters to evaluate LRA against other explainers
- We assume that images are normalized to the [-1, 1] range - this may need some adjustments when connecting to Quantus metrics

For ResNet50:
- Relevance Pullbacks achieve ~100% Response Faithfulness (RF), while standard gradients have ~45% RF
- Semantic Pullbacks (LRA with 5 steps, each of L2 length 20) achieve ~80% RF, while standard PGA has ~35% RF


# TODO

- choose temperatures maximising the RF estimate - thus defining Relevance Pullbacks
- evaluate the response faithfulness of other explainers
- evaluate relevance pullbacks in Quantus
- enhance adversarial perturbations toward adversarial (predicted) class