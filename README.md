# Thesis Research Repository

Welcome to the research repository for our thesis on the application of distributionally robust optimization models in the context of wind/hydrogen power plants. This repository contains the models implemented for analysis and comparison.

## Data
Due to confidentiality reasons, the actual data used in the thesis cannot be shared in this repository. However, we have provided some dummy data for which we cannot guarantee the same performance as reported in the thesis. The data includes the following:

* Day ahead prices for DK1, source: Energi Data Service 
* Imbalance prices for DK1, source: ENTSO-E
* Wind power and wind power forecasts for DK1, source: ENTSO-E
* Load forecast for DK1, source: ENTSO-E
* Shedueled flow on transmission lines for DK1, source: ENTSO-E

Please note that the price forecasts are generated by adding t-distributed noise to the realized prices.

## Models
This repository includes the implementation of several models used in the thesis. The models can be found in the models directory and are organized as follows:

* Deterministic Model: 
  * This model is based on the approach described in the paper: [link to paper](https://arxiv.org/abs/2301.05310).
* Sample Average Approximation (SAA) Model:
  * This model uses a sample average approximation method to address uncertainty in the data.
* Wasserstein-based Distributionally Robust Optimization (DRO) Model:
  * This model incorporates the Wasserstein distance metric to handle uncertainty and robustness considerations.
* Wasserstein-based DRO Model with Covariance Structure:
  * This model extends the previous DRO model by including the covariance structure between the imbalance and day ahead prices in the ambiguity set. The model is a mixed-integer SDP and to solve the problem you will nedd the solver [pajarito](https://github.com/jump-dev/Pajarito.jl).
* Two-Stage Wasserstein DRO Model:
  * This model is equivalent to a two-stage sample average approximation model but formulated within the DRO framework. When the Wasserstein radius is set to 0, it reduces to a standard two-stage SAA model.

All 1-stage models use a input file called input-1stage.jl. It will setup all relevant constants and errors for the error sets for each hour of the optimization problems. For the 2-stage problem there is a similar file called input-2stage.jl.

## Authors
This thesis research is conducted by Anton Ruby Larsen and Mads Esben Hansen. 
