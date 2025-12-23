# NOTES

- We don't implement the directional faithfulness as Quantus metric - because Quantus enforces the casting to cpu and numpy, which is awkward. Instad, we compute directional faithfulness in Pytorch, using attribution methods straight from Captum (especially since Quantus is a simple wrapper for Captum).
- But we do use Quantus adapters to evaluate LRA against other explainers
- We assume that images are normalized to the [-1, 1] range - this may need some adjustments when connecting to Quantus metrics