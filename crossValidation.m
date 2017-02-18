function[percentError] = crossValidation(net,foldNumber,excelName)
%% Reads Excel File
trainingfileName = excelName;
[trainNumeric,text,excel] = xlsread(trainingfileName);
folds = foldNumber;
error = zeros(folds,1);
correct = zeros(folds,1);
%% Randomly Sort Data
% Generate random numbers from 1 to number of training examples
randomPerms = randperm(size(trainNumeric,1)).';
% Combine the random column vector with the normal data
trainNumeric = [randomPerms,trainNumeric];
% Sort the matrix based on the random number
trainNumeric = sortrows(trainNumeric);
% Remove the random numbers in the first column
trainNumeric = trainNumeric(:,2:size(trainNumeric,2)).';

%% Cross-Validation
% Rounds fold size to the nearest integer so that all folds all the same
foldSize = floor(size(trainNumeric,2) / folds);
startIndex = 1;
endIndex = foldSize;
% Check to see if folds are correct
foldCheck = struct('TestFolds',0,'TrainFolds',0);
for numberOfFoldsRan = 1:folds
    testFold = trainNumeric(:,startIndex:endIndex);
    % Combines all the folds after the one test fold
    if startIndex == 1
        trainFold = trainNumeric(:,endIndex + 1:size(trainNumeric,2));
    % Combines all the folds except the one test fold
    else
        trainFold = cat(2,trainNumeric(:,1:startIndex-1),trainNumeric(:,endIndex+1:size(trainNumeric,1)));
    end
    startIndex = startIndex + foldSize;
    endIndex = endIndex + foldSize;
    foldCheck(numberOfFoldsRan).TestFolds = testFold;
    foldCheck(numberOfFoldsRan).TrainFolds = trainFold;
    % Divide the folds into their data and their known output
    trainFoldTargetVec = trainFold(1,:);
    testFoldTargetVec = testFold(1,:);
    trainFoldData = trainFold(2:size(trainFold,1),:);
    testFoldData = testFold(2:size(testFold,1),:);
    % Train Network
    [net,record] = train(net,trainFoldData,trainFoldTargetVec,'useGPU','yes');
    error(numberOfFoldsRan,1) = perform(net,testFoldData,testFoldTargetVec);
    % Calculate Percent Correct
    output = net(testFoldData);
    binOutput = output > .50;
    numCorrect = find(binOutput == testFoldTargetVec);
    percentCorrect = size(numCorrect,2) / size(testFoldTargetVec,2);
    correct(numberOfFoldsRan,1) = percentCorrect;
end
percentError = mean(correct);