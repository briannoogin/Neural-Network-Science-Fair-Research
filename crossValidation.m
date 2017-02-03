function[percentError] = crossValidation(net,foldNumber)
%% Reads Excel File
trainingfileName = 'breastCancerFullDataSet';
[trainNumeric,text,excel] = xlsread(trainingfileName);
folds = foldNumber;
error = zeros(foldNumber,1);
trainTargetVector = trainNumeric(:,1);
%% Randomly Sort Data
% Generate random numbers from 1 to number of training examples
randomPerms = randperm(size(trainNumeric,1)).';
% Combine the random column vector with the normal data
trainNumeric = [randomPerms,trainNumeric];
% Sort the matrix based on the random number
trainNumeric = sortrows(trainNumeric);
% Remove the random numbers in the first column
trainNumeric = trainNumeric(:,2:size(trainNumeric,2));

%% Cross-Validation
% Rounds fold size to the nearest integer so that all folds all the same
foldSize = floor(size(trainNumeric,1) / folds);
startIndex = 1;
endIndex = foldSize;
for numberOfFoldsRan = 1:folds
    testFold = trainNumeric(startIndex:endIndex,:);
    % Combines all the folds after the one test fold
    if startIndex == 1
        trainFold = trainNumeric(endIndex + 1:size(trainNumeric,1),:);
    % Combines all the folds except the one test fold
    else
        trainFold = cat(1,trainNumeric(1:startIndex,:),trainNumeric(endIndex:size(trainNumeric,1),:));
    end
    startIndex = startIndex + foldSize;
    endIndex = endIndex + foldSize;
    % Add the variables 
    trainFold = cat(1,text,num2cell(trainFold));
    testFold = cat(1,text,num2cell(testFold));
    % Train Network
    [net,record] = train(net,trainNumeric,trainTargetVector,'useGPU','yes');
    error(folds) = net(testFold);
end
percentError = mean(error);