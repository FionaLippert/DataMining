### important things to consider regarding the position estimate
- don't use the training data that has been used for training the position model for training the main ranking model!
- use the feature 'position_estimate' for both training and validation, and not the original 'position'

### important things to consider regarding the 'prop_desirability' feature
- use 'prop_desirability' for training and 'prop_desirability_incomplete' for validation (to mimic performance on the test set where for some properties the information is not available)

### target variables
- 'booking_bool'
- 'click_bool'
- 'booked_clicked_combined'
