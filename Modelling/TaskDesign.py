from random import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import numpy.ma as ma
import pickle
import seaborn as sns


class task_Design:
    def __init__(
        self,
        mainTrials,
        additionalTrials,
        alpha=None,
        beta=None,
        alpha2=None,
        alphaOther=None,
        alphaOther2=None,
        alphaOther3=None,
        alphaOther4=None,
        V_option0Init=None,
        V_option1Init=None,
        pearce=0,
        omega=None,
    ):
        """How task is conducted"""

        self.mainTrials = mainTrials
        self.additionalTrials = additionalTrials
        self.greenCells = (0, 0), (1, 1), (2, 2)

        self.modality0A = 0.5
        self.modality0B = 0.35
        self.modality0C = 0.15
        self.modality1A = 0.15
        self.modality1B = 0.50
        self.modality1C = 0.35
        self.modality2A = 0.35
        self.modality2B = 0.15
        self.modality2C = 0.50

        self.rewardProb = 0.8
        self.rewardMag = 1.0
        self.punishProb = 0.0
        self.punishMag = 0.0

        if pearce == 1:
            self.omegaPearce = np.empty(self.mainTrials)
            self.omega2Pearce = np.empty(self.mainTrials)

        ## Making vectors of reward with 0.8 probability
        tempFeedbackAccuracy = np.repeat(
            np.array([1, 0]),
            np.array([round(self.rewardProb * 10), round((1 - self.rewardProb) * 10)]),
        )
        tempFeedbackAccuracy = tempFeedbackAccuracy[
            np.random.permutation(np.size(tempFeedbackAccuracy))
        ]

        self.feedbackAccuracy = np.empty(0)

        for i in range(int(mainTrials / (np.size(tempFeedbackAccuracy)))):
            self.feedbackAccuracy = np.concatenate(
                (
                    self.feedbackAccuracy,
                    (
                        tempFeedbackAccuracy[
                            np.random.permutation(np.size(tempFeedbackAccuracy))
                        ]
                    ),
                ),
                axis=0,
            )

        """ How participants learn : Recoverable parameters """
        self.pearce = pearce

        if alpha is None:
            self.alpha = np.random.uniform(0, 1)
        else:
            self.alpha = alpha

        if beta is None:
            self.beta = 0 + 15.0 * random()
        else:
            self.beta = beta  # 2.0

        self.alpha2 = alpha2
        self.alphaOther = alphaOther
        self.alphaOther2 = alphaOther2
        self.alphaOther3 = alphaOther3
        self.alphaOther4 = alphaOther4

        if self.pearce == 1:
            self.omega = 1
            self.omega2 = 1

        self.statLearnPar = 1  # Bayesian parameter

        self.V_option0 = np.empty((self.mainTrials + 1, 3, 3))
        self.V_option0[:] = np.nan
        if V_option0Init is None:
            self.V_option0[0, :] = 0.5
        else:
            self.V_option0[0, :] = V_option0Init
        self.V_option1 = np.empty((self.mainTrials + 1, 3, 3))
        self.V_option1[:] = np.nan
        if V_option1Init is None:
            self.V_option1[0, :] = 0.5
        else:
            self.V_option1[0, :] = V_option1Init

        self.statCount = np.zeros((self.mainTrials + self.additionalTrials + 1, 3, 3))
        self.statSurprise = np.empty((self.mainTrials + self.additionalTrials, 3, 3))
        self.statSurprise[:] = np.nan
        self.statSurpriseRow = np.empty((self.mainTrials + self.additionalTrials, 3, 3))
        self.statSurpriseRow[:] = np.nan
        self.statSurpriseColumn = np.empty(
            (self.mainTrials + self.additionalTrials, 3, 3)
        )
        self.statSurpriseColumn[:] = np.nan

        self.rowBeliefs = np.empty((self.mainTrials + self.additionalTrials + 1, 3, 3))
        self.rowBeliefs[:] = np.nan
        self.columnBeliefs = np.empty(
            (self.mainTrials + self.additionalTrials + 1, 3, 3)
        )
        self.columnBeliefs[:] = np.nan
        self.beliefsStat = np.empty((self.mainTrials + self.additionalTrials + 1, 3, 3))
        self.beliefsStat[:] = np.nan
        self.stimulusPair = np.empty((self.mainTrials + self.additionalTrials, 2))
        self.stimulusPair[:] = np.nan
        self.correctResponse = np.empty((self.mainTrials, 1))
        self.reward = np.empty((self.mainTrials, 3, 3))
        self.reward[:] = np.nan
        self.rewardPE = np.empty((self.mainTrials, 3, 3))
        self.rewardPE[:] = np.nan
        self.action = np.empty((self.mainTrials, 1))
        self.action[:] = np.nan
        self.choiceProb = np.empty((self.mainTrials, 2))
        self.choiceProb[:] = np.nan
        self.reshapedV_option0 = np.empty((self.mainTrials + 1, 9))
        self.accurate = np.empty((self.mainTrials, 1))
        self.errorPercentage = np.nan
        self.simulatedData = np.empty((self.mainTrials + self.additionalTrials, 4))
        self.simulatedData[:] = np.nan

    """ Defining the task strcture"""

    def taskStructure(self, taskStruct=None, green=None, feedbackAcc=None):
        if taskStruct is None:
            self.taskStruct = np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            )
            self.taskStruct = np.repeat(
                self.taskStruct,
                (
                    (self.mainTrials + self.additionalTrials)
                    / 3.0
                    * np.array(
                        [
                            self.modality0A,
                            self.modality0B,
                            self.modality0C,
                            self.modality1A,
                            self.modality1B,
                            self.modality1C,
                            self.modality2A,
                            self.modality2B,
                            self.modality2C,
                        ]
                    )
                ).astype(int),
                axis=0,
            )
            self.randomLayout = np.random.permutation(
                self.mainTrials + self.additionalTrials
            )
            self.taskStruct = self.taskStruct[self.randomLayout]
        else:
            self.taskStruct = taskStruct
            self.greenCells = green[0], green[1], green[2]
            self.feedbackAccuracy = feedbackAcc

        for i in range(0, self.mainTrials + self.additionalTrials):
            visual = self.taskStruct[i, 0]
            audio = self.taskStruct[i, 1]
            self.stimulusPair[i, :] = visual, audio
            self.stimulusPair = self.stimulusPair.astype(int)
            self.statCount[i + 1, :] = self.statCount[i, :]
            self.statCount[(i + 1,) + tuple(self.stimulusPair[i, :])] = (
                self.statCount[(i,) + tuple(self.stimulusPair[i, :])] + 1
            )

            self.simulatedData[i, 0] = self.stimulusPair[i, 0]
            self.simulatedData[i, 1] = self.stimulusPair[i, 1]

        # print("stimulus", self.stimulusPair)
        # print(self.statCount)

    """RL"""

    def RLloops(self):
        error = 0
        # Trials starting after the additional trials first
        for i in range(0, self.mainTrials):
            """Actions taken"""

            self.choiceProb[i, 0] = np.exp(
                self.beta
                * self.V_option0[
                    ((i,) + tuple(self.stimulusPair[i + self.additionalTrials, :]))
                ]
            ) / (
                (
                    np.exp(
                        self.beta
                        * self.V_option0[
                            (
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            )
                        ]
                    )
                )
                + (
                    np.exp(
                        self.beta
                        * self.V_option1[
                            (
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            )
                        ]
                    )
                )
            )
            self.choiceProb[i, 1] = 1 - self.choiceProb[i, 0]
            rand3 = random()
            if rand3 < self.choiceProb[i, 0]:
                self.action[i] = 0
            else:
                self.action[i] = 1

            """defining reward"""

            self.reward[i, :] = 0
            if (
                tuple(self.stimulusPair[i + self.additionalTrials, :])
                in self.greenCells
            ):  # When in green cells.
                self.correctResponse[i] = 0
                if (
                    self.action[i] == self.correctResponse[i]
                ):  # correct action for green cells
                    self.accurate[i] = 1
                    self.reward[
                        (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ] = self.feedbackAccuracy[i]

                else:  # incorrect action for greenCells
                    error = error + 1
                    self.accurate[i] = 0
                    self.reward[
                        (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ] = (1 - self.feedbackAccuracy[i])

            else:  # When in white cells
                self.correctResponse[i] = 1
                if (
                    self.action[i] == self.correctResponse[i]
                ):  # correct action for whiteCells (non green cells)
                    self.accurate[i] = 1
                    self.reward[
                        (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ] = self.feedbackAccuracy[i]

                else:  # Incorrect action for whiteCells (non green cells)
                    error = error + 1
                    self.accurate[i] = 0
                    self.reward[
                        (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ] = (1 - self.feedbackAccuracy[i])

            """updating values"""

            self.rewardPE[i, :] = np.nan
            SimOtherPairs = [
                p
                for p in list(np.unique(self.taskStruct, axis=0))
                if bool(p[0] == self.stimulusPair[i, 0])
                ^ bool(p[1] == self.stimulusPair[i, 1])
            ]

            if self.action[i] == 0:
                self.rewardPE[
                    (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                ] = (
                    self.reward[
                        (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ]
                    - self.V_option0[
                        (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ]
                )
                self.V_option0[i + 1, :] = self.V_option0[i, :]
                if self.pearce == 1:
                    self.omega = (
                        self.omega
                        + (
                            abs(
                                self.rewardPE[
                                    (i,)
                                    + tuple(
                                        self.stimulusPair[i + self.additionalTrials, :]
                                    )
                                ]
                            )
                            - self.omega
                        )
                        * self.alpha
                    )
                    self.V_option0[
                        (i + 1,)
                        + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ] = (
                        self.V_option0[
                            (i,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ]
                        + self.omega
                        * self.rewardPE[
                            (i,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ]
                    )
                else:
                    self.V_option0[
                        (i + 1,)
                        + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ] = (
                        self.V_option0[
                            (i,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ]
                        + self.alpha
                        * self.rewardPE[
                            (i,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ]
                    )
                if self.alphaOther3:
                    if (
                        self.reward[
                            (i,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ]
                        == 1
                    ):
                        for p2 in SimOtherPairs:
                            self.V_option0[(i + 1,) + tuple(p2)] = self.V_option0[
                                (i,) + tuple(p2)
                            ] + self.alphaOther * (
                                1
                                - self.reward[
                                    (i,)
                                    + tuple(
                                        self.stimulusPair[i + self.additionalTrials, :]
                                    )
                                ]
                                - self.V_option0[(i,) + tuple(p2)]
                            )
                    else:
                        for p2 in SimOtherPairs:
                            self.V_option0[(i + 1,) + tuple(p2)] = self.V_option0[
                                (i,) + tuple(p2)
                            ] + self.alphaOther2 * (
                                1
                                - self.reward[
                                    (i,)
                                    + tuple(
                                        self.stimulusPair[i + self.additionalTrials, :]
                                    )
                                ]
                                - self.V_option0[(i,) + tuple(p2)]
                            )
                elif self.alphaOther and not self.alphaOther3:
                    for p in SimOtherPairs:
                        self.V_option0[(i + 1,) + tuple(p)] = self.V_option0[
                            (i,) + tuple(p)
                        ] + self.alphaOther * (
                            1
                            - self.reward[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                            - self.V_option0[(i,) + tuple(p)]
                        )
                self.V_option1[i + 1, :] = self.V_option1[i, :]
            else:
                self.rewardPE[
                    (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                ] = (
                    self.reward[
                        (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ]
                    - self.V_option1[
                        (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
                    ]
                )
                self.V_option1[i + 1, :] = self.V_option1[i, :]
                if self.alpha2:
                    if self.pearce == 1:
                        self.omega2 = (
                            self.omega2
                            + (
                                abs(
                                    self.rewardPE[
                                        (i,)
                                        + tuple(
                                            self.stimulusPair[i + self.additionalTrials, :]
                                        )
                                    ]
                                )
                                - self.omega2
                            )
                            * self.alpha2
                        )
                        self.V_option1[
                            (i + 1,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ] = (
                            self.V_option1[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                            + self.omega2
                            * self.rewardPE[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                        )
                    else:
                        self.V_option1[
                            (i + 1,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ] = (
                            self.V_option1[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                            + self.alpha2
                            * self.rewardPE[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                        )
                else:
                    if self.pearce == 1:
                        self.omega = (
                            self.omega
                            + (
                                abs(
                                    self.rewardPE[
                                        (i,)
                                        + tuple(
                                            self.stimulusPair[i + self.additionalTrials, :]
                                        )
                                    ]
                                )
                                - self.omega
                            )
                            * self.alpha
                        )
                        self.V_option1[
                            (i + 1,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ] = (
                            self.V_option1[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                            + self.omega
                            * self.rewardPE[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                        )
                    else:
                        self.V_option1[
                            (i + 1,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ] = (
                            self.V_option1[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                            + self.alpha
                            * self.rewardPE[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                        )

                if self.alphaOther and not self.alphaOther2:
                    for p3 in SimOtherPairs:
                        self.V_option1[(i + 1,) + tuple(p3)] = self.V_option1[
                            (i,) + tuple(p3)
                        ] + self.alphaOther * (
                            1
                            - self.reward[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                            - self.V_option1[(i,) + tuple(p3)]
                        )
                elif self.alphaOther2 and not self.alphaOther3:
                    for p3 in SimOtherPairs:
                        self.V_option1[(i + 1,) + tuple(p3)] = self.V_option1[
                            (i,) + tuple(p3)
                        ] + self.alphaOther2 * (
                            1
                            - self.reward[
                                (i,)
                                + tuple(self.stimulusPair[i + self.additionalTrials, :])
                            ]
                            - self.V_option1[(i,) + tuple(p3)]
                        )
                elif self.alphaOther3:
                    if (
                        self.reward[
                            (i,)
                            + tuple(self.stimulusPair[i + self.additionalTrials, :])
                        ]
                        == 1
                    ):
                        for p3 in SimOtherPairs:
                            self.V_option1[(i + 1,) + tuple(p3)] = self.V_option1[
                                (i,) + tuple(p3)
                            ] + self.alphaOther3 * (
                                1
                                - self.reward[
                                    (i,)
                                    + tuple(
                                        self.stimulusPair[i + self.additionalTrials, :]
                                    )
                                ]
                                - self.V_option1[(i,) + tuple(p3)]
                            )
                    else:
                        for p3 in SimOtherPairs:
                            self.V_option1[(i + 1,) + tuple(p3)] = self.V_option1[
                                (i,) + tuple(p3)
                            ] + self.alphaOther4 * (
                                1
                                - self.reward[
                                    (i,)
                                    + tuple(
                                        self.stimulusPair[i + self.additionalTrials, :]
                                    )
                                ]
                                - self.V_option1[(i,) + tuple(p3)]
                            )
                self.V_option0[i + 1, :] = self.V_option0[i, :]

            self.simulatedData[i + self.additionalTrials, 2] = self.rewardPE[
                (i,) + tuple(self.stimulusPair[i + self.additionalTrials, :])
            ]

            if self.pearce:
                self.alphaPearce[i] = self.alpha

        self.errorPercentage = (error / self.mainTrials) * 100

    # print("action", self.action)
    # print("reward", self.reward)
    # print(self.rewardPE)
    # print("V_option0", self.V_option0)
    # print("V_option1", self.V_option1)

    """ Statistical learning """

    def statisticalLearning(self):
        for i in range(0, self.mainTrials + self.additionalTrials + 1):
            if i == 0:
                rowDen0 = 3 * self.statLearnPar
                rowDen1 = 3 * self.statLearnPar
                rowDen2 = 3 * self.statLearnPar
                columnDen0 = 3 * self.statLearnPar
                columnDen1 = 3 * self.statLearnPar
                columnDen2 = 3 * self.statLearnPar
                self.rowBeliefs[i, :] = (
                    (self.statLearnPar + self.statCount[i, :]) / 3 * self.statLearnPar
                )
                self.columnBeliefs[i, :] = (
                    (self.statLearnPar + self.statCount[i, :]) / 3 * self.statLearnPar
                )
                num = self.statLearnPar + self.statCount[i, :]
                den = 9 * self.statLearnPar
                # Total statistical beliefs irrespective of rows and columns.
                self.beliefsStat[i, :] = (num) / (den)
            else:
                self.rowBeliefs[i, :] = self.rowBeliefs[i - 1, :]
                # Row beliefs
                if self.simulatedData[i - 1, 0] == 0:
                    rowDen0 = rowDen0 + 1
                    self.rowBeliefs[i, 0, :] = (
                        self.statLearnPar + self.statCount[i, 0, :]
                    ) / rowDen0
                elif self.simulatedData[i - 1, 0] == 1:
                    rowDen1 = rowDen1 + 1
                    self.rowBeliefs[i, 1, :] = (
                        self.statLearnPar + self.statCount[i, 1, :]
                    ) / rowDen1
                else:
                    rowDen2 = rowDen2 + 1
                    self.rowBeliefs[i, 2, :] = (
                        self.statLearnPar + self.statCount[i, 2, :]
                    ) / rowDen2
                # column beliefs
                self.columnBeliefs[i, :] = self.columnBeliefs[i - 1, :]
                if self.simulatedData[i - 1, 1] == 0:
                    columnDen0 = columnDen0 + 1
                    self.columnBeliefs[i, 0, :] = (
                        self.statLearnPar + self.statCount[i, 0, :]
                    ) / columnDen0
                elif self.simulatedData[i - 1, 1] == 1:
                    columnDen1 = columnDen1 + 1
                    self.columnBeliefs[i, 1, :] = (
                        self.statLearnPar + self.statCount[i, 1, :]
                    ) / columnDen1
                else:
                    columnDen2 = columnDen2 + 1
                    self.columnBeliefs[i, 2, :] = (
                        self.statLearnPar + self.statCount[i, 2, :]
                    ) / columnDen2
                num = self.statLearnPar + self.statCount[i, :]
                den = den + 1
                # Total statistical beliefs irrespective of rows and columns.
                self.beliefsStat[i, :] = (num) / (den)

        # Surprises calculated from beliefs update
        for i in range(0, self.mainTrials + self.additionalTrials):
            self.statSurpriseRow[i, :] = np.nan
            self.statSurpriseRow[(i,) + tuple(self.stimulusPair[i, :])] = -np.log(
                self.rowBeliefs[(i,) + tuple(self.stimulusPair[i, :])]
            )
            self.statSurpriseColumn[i, :] = np.nan
            self.statSurpriseColumn[(i,) + tuple(self.stimulusPair[i, :])] = -np.log(
                self.columnBeliefs[(i,) + tuple(self.stimulusPair[i, :])]
            )
            self.statSurprise[i, :] = np.nan
            self.statSurprise[(i,) + tuple(self.stimulusPair[i, :])] = -np.log(
                self.beliefsStat[(i,) + tuple(self.stimulusPair[i, :])]
            )

            self.simulatedData[i, 3] = self.statSurprise[
                (i,) + tuple(self.stimulusPair[i, :])
            ]
