% Reads Excel File
trainingfileName = 'scienceFairBreastCancerData.xls';
[trainNumeric,text,excel] = xlsread(trainingfileName);
testfileName = 'breastTestData.xlsx';
[testNumeric,testText,testExcel] = xlsread(testfileName);
% Gets the data targets
trainTargetVector = trainNumeric(:,1);
testTargetVector = testNumeric(:,1);
% Gets the data
trainingInputMatrix = trainNumeric(:,2:size(excel,2));
testInputMatrix = testNumeric(:,2:size(excel,2));
% Creates network with three layers
net = feedforwardnet(10);
net.numlayers = 4;
% Tranpose data so it works with the toolbox
trainTargetVector = trainTargetVector.';
trainingInputMatrix = trainingInputMatrix.';
testTargetVector = testTargetVector.';
testInputMatrix = testInputMatrix.';
% Name the layers
net.layers{1}.name = 'Hidden Layer 1';
net.layers{2}.name = 'Hidden Layer 2';
net.layers{3}.name = 'Hidden Layer 3';
net.layers{4}.name = 'Output';
% Designate the activation functions and number of neural units
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'logsig';
net.layers{4}.transferFcn = 'softmax';
net.layers{1,1}.size = 1000;
net.layers{2,1}.size = 500;
net.layers{3,1}.size = 250;
% Connects layers and bias units
net.layerConnect = [0 0 0 0; 1 0 0 0; 0 1 0 0;0 0 1 0];
net.biasConnect = [1;1;1;1];
net.outputConnect = [0 0 0 1];
% Shape the network to the data
%net = configure(net,trainingInputMatrix,trainTargetVector);
net.performParam.regularization = 0;
net.trainFcn = 'trainscg';
net.initFcn = 'initlay';
net.layers{1}.netInputFcn = 'netsum';
net.layers{2}.netInputFcn = 'netsum';
net.layers{3}.netInputFcn = 'netsum';
% Change train settings and train network 
net.trainParam.showWindow = 0;
trialPercentPerformance = zeros(10,1);
trialTrainPerformance = zeros(10,1);
% Do multiple trials for percent correct
performanceStruct = struct('Net',0,'Output',0,'Xcoord',0,'Ycoord',0,'AUC',0);
for trial = 1 : 10
net = init(net);
[net1,record] = train(net,trainingInputMatrix,trainTargetVector,'useGPU','yes');
trainPerformance = perform(net,trainingInputMatrix,trainTargetVector);
% Calculating percent correct
y = net1(trainingInputMatrix);
%y = y > .50;
findCorrect = find(y == trainTargetVector);
percentCorrect = size(findCorrect,2) / size(trainTargetVector,2); 
trialPercentPerformance(trial) = percentCorrect;
trialTrainPerformance(trial) = trainPerformance;
[xCoordinate,yCoordinate,threshhold,AUC] = perfcurve(trainTargetVector,y,1);
performanceStruct(trial).Net = net1;
performanceStruct(trial).Output = y;
performanceStruct(trial).Xcoord = xCoordinate;
performanceStruct(trial).Ycoord = yCoordinate;
performanceStruct(trial).AUC = AUC;
net.performParam.regularization = net.performParam.regularization + .1;
end
[~,bestPerformanceIndex] = max(trialPercentPerformance);
net = performanceStruct(bestPerformanceIndex).Net;
% Calculate the average performance
percentCorrect = sum(trialPercentPerformance,1) / size(trialPercentPerformance,1);
trainPerformance = sum(trialPercentPerformance,1) / size(trialPercentPerformance,1);
% Test Performance
testPerformance = perform(net,testInputMatrix,testTargetVector);

% Finds where the network has failed
error = find(~y == trainTargetVector);
% Finds all the errors and collects them in a matrix
errorMatrix = cell(size(error,2) + 1,size(excel,2));
errorMatrix(1,:) = excel(1,:);
trainNumeric = num2cell(trainNumeric);
for errorIndex = 1 : size(error,2)
    errorMatrix(errorIndex + 1,:) = trainNumeric(error(1,errorIndex),:);
end
% ROC Curve
numberOfTruePositives = size(find(y == trainTargetVector & trainTargetVector == 1),2);
numberOfTrueNegatives = size(find(y == trainTargetVector & trainTargetVector == 0),2);
numberOfFalseNegatives = size(find(y ~= trainTargetVector & trainTargetVector == 1),2);
numberOfFalsePositives = size(find(y ~= trainTargetVector & trainTargetVector == 0),2);
sensitivity = numberOfTruePositives / (numberOfTruePositives + numberOfFalseNegatives); 
specificity = numberOfTrueNegatives / (numberOfTrueNegatives + numberOfFalsePositives); 
hold on
title('Regularization ROC Curve');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('0.0 Regularization','0.1 Regularization','0.2 Regularization','0.3 Regularization','0.4 Regularization',...
    '0.5 Regularization','0.6 Regularization','0.7 Regularization','0.8 Regularization','0.9 Regularization',...
    '1.0 Regularization')

for plotIndex = 1 : trial
plot(performanceStruct(plotIndex).Xcoord,performanceStruct(plotIndex).Ycoord);
end