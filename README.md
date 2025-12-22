# NOTES

- We don't implement the directional faithfulness as Quantus metric - as Quantus enforces the casting to cpu and numpy, which is awkward. Instad, we compute directional faithfulness in Pytorch, using attribution methods straight from Captum.
- But we do use Quantus adapters to evaluate LRA against other explainers
- We assume that images are normalized to the [-1, 1] range