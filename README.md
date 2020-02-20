# Implementation-of-binary-Ncut-to-predict-politician-party

I used the dataset from THOMAS, a website of votes for political stands. Politicians vote for bills on that website. The vote can be binary (1 or -1) represents whether they like a bill or no. In the associated dataset, you will find two *.pkl files that contains:

- The count of user A and user B both votes for a same bill with same attitude as 1.
- The count of user A and user B both votes for a same bill with same attitude as -1.

Using the covote relationship, I partitioned the politicians by the Ncut algorithm and find who belongs to whic Party - Demographic or Republican. The ground truth is given in Dict_Person_Senate and Dict_Person_House.

The result for the implementation is not too great, because in the data, many politicians have none extreme viewpoints, meaning they vote Democratically for some and Republicanly for other. This causes the accuracy to be 0.5
