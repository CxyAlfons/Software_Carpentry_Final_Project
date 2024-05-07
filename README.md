1. Introduction

    This python script uses different machine learning models to learn the amino acid sequences of antimicrobial peptides (AMPs). The trained models are used to predict the antimicrobial activities and their prediction performances are compared to each other. And the ones with best performances are also compared to the true values.

2. Data

    The data used in this analysis includes active AMPs (with at least minimum antimicrobial activities) and also inactive AMPs. The active AMP data is sourced from Gull, S., & Minhas, F. (2022) along with the corresponding minimum inhibitory concentration (MIC) data. To mitigate the bias caused by data setup (only active AMPs are preserved to have effective MIC), inactive AMP data is also included and sourced from Li, C. et al. (2022).
    
    Since inactive AMPs have no MIC data and their MIC values should theoretically be positive infinity, a transformation is performed to change MIC values into activity score: activity score = MIC / (MIC + 128).Such transformation changes all MIC into values ranging from 0 to 1 with inactive AMPs having score of 1. 128 is a coefficient, which is reported to be the MIC boundary of lower active AMPs (Richter, A. et al. (2022))

    The data are stored in the repository as "AMP0_data.csv" (active AMPs) and "AMPlify_non_AMP_imbalanced.fa" (inactive AMPs).

3. Model

    The following models are used in this analysis:
    - Linear Regression
    - Ridge
    - Lasso
    - Random Forest
    - SVR
    - kNN
    - Multilayer Perceptron Neural Network
    - Convolution Recurrent Neural Network

4. Products

    The analysis (comparison of the different models) are visualized as plots and are stored in the repository as well, including the distribution of original data and the performance of different models. To regenerate the plots, please download the whole repository and run the "Learning.py".

5. References

    Gull, S., & Minhas, F. (2022). AMP0: Species-Specific Prediction of Anti-microbial Peptides Using Zero and Few Shot Learning. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 19(1), 275–283. https://doi.org/10.1109/TCBB.2020.2999399

    Li, C., Sutherland, D., Hammond, S. A., Yang, C., Taho, F., Bergman, L., Houston, S., Warren, R. L., Wong, T., Hoang, L. M. N., Cameron, C. E., Helbing, C. C., & Birol, I. (2022). AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides effective against WHO priority pathogens. BMC Genomics, 23(1), 77–77. https://doi.org/10.1186/s12864-022-08310-4

    Richter, A., Sutherland, D., Ebrahimikondori, H., Babcock, A., Louie, N., Li, C., Coombe, L., Lin, D., Warren, R. L., Yanai, A., Kotkoff, M., Helbing, C. C., Hof, F., Hoang, L. M. N., & Birol, I. (2022). Associating Biological Activity and Predicted Structure of Antimicrobial Peptides from Amphibians and Insects. Antibiotics (Basel), 11(12), 1710-. https://doi.org/10.3390/antibiotics11121710
    