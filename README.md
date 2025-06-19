# Bot Detection in Email Engagement  
**Moein Izadi | Merck Data Science Co-op Project**

This repository documents a supervised machine learning approach to detect inorganic email engagement (bot opens and clicks) in healthcare marketing. The work builds on internal business-rule-based systems and honeypot link deployment strategies developed at Merck Germany and the Netherlands. The final solution integrates labeled data, extensive EDA, time-based feature engineering, and classification modeling.

## Project Overview

- **Context:** Inorganic engagement undermines the accuracy of customer journey analytics. This project addresses this using labeled data and predictive ML.
- **Goal:** Automatically identify bot behavior in email click/open data using labeled honeypot activity and engineered time features.
- **Tools Used:** Python (PyCaret, scikit-learn), AWS SageMaker, Jupyter Notebooks, HTML exports

## Key Contributions

- Integrated multiple labeling strategies (business rules + honeypots)
- Engineered time-differential features (e.g., Sent-to-Open, Sent-to-Click)
- Evaluated 13+ ML models with cross-validation and stratified splits
- Investigated and mitigated model leakage risks
- Deployed models with >95% AUC on honeypot-only validation data

## Reports (Interactive)
- [Full Article (report)](./Merck/Detection%20of%20Bot%20inorganic%20engagement%20on%20marketing%20emails%20using.pdf)




## Licensing

This repository is licensed under the [MIT License](./LICENSE). You may use this code and methodology with attribution.

## Citation

> Izadi, M., Giri, S., Brown, J., Shi, C. (2022). *Detection of Bot Inorganic Engagement on Marketing Emails Using Supervised Machine Learning Algorithms*. Merck Data Science & Analytics Solutions.
