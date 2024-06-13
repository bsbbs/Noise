- TrnsfrmData.csv and TrnsfrmData.mat contain the same set of behavioral dataset from 60 Craigslist subjects we collected at NYU until Nov 2022, in format of Matlab and text respectively.
- CorrectItems.cvs contains every item's ID, mapped with the detailed information of the amazon goods.

A list of variables in TrnsfrmData:
- Group: The randomization of each subject, that determines which half of the items were precise (definitive) to them and the other half items are vague (ambiguous) to them.
- subID: the subject ID with the information of collecting date.
- Trial: the number of trials in the choice task, in total 360 trials, with 2*2 conditions (see below variables)
- TimePressure: time pressure for choice, High = 1.5s maximum and Low = 10 s maximum.
- TimeConstraint: the actually value of time constraints, the same meaning as specified above.
- Vagueness: The vagueness of distractor (V3). In the design, only the lower value distractor was set as vague or precise. The target V1 and V2 were always precise. Mean bid of V2 is equal or higher than V1.
- Vaguenesscode: 1 for vague and 0 for precise
- ID1, ID2, and ID3: the item ID for the options 1, 2, and 3.
- V1, V2, and, V3: the mean bid value of options 1, 2, and 3.
- ID_left, ID_middle, ID_right: the item ID appeared on the specific position of the screen, since the positions for options are randomized trial-by-trial.
- Bid_left, Bid_middle, Bid_right: similar to above, the mean bid values for the item presented on specific positions of the screen.
- chosenItem: the chosen option, 1, 2, or 3
- RT: reaction time for choice
- RTvariance: uncertainty of measurement on RT because of hardware limitation
- chosenPosition: choice recorded as the position of the screen, instead of the item or option
- chosenID: choice recorded as the ID of the chosen item
- chosenValue: the bid mean of the chosen item.
- fixation_time, option_time, choice_time, feedback_time: recorded time onsets for the events in a trial: fixation cross (ITI), onset of the options, participant's response time, and chosen feedback with the chosen item squared in red.
- timeout: whether the participant ran out of time for choice
- Definitive: opposite to vaguenesscode, 0 is vague distractor and 1 is precise distractor.
- sdV1, sdV2, and sdV3: the bid variance of each options over three times.


A list of variables in CorrectItems:
Items: ID of each item
Name: short name of the item
Actual name of item on website
Patch: The patch that every item belongs to. The item is a precise item to a subject if the patch ID equals to the group ID of the subject.
