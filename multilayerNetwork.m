% Reads Excel File
trainingfileName = 'mammographBreastCancerWithDeletedData';
[trainNumeric,text,excel] = xlsread(trainingfileName);
testfileName = 'mammographBreastCancerWithDeletedData';
[testNumeric,testText,testExcel] = xlsread(testfileName);
% Gets the data targets
trainTargetVector = trainNumeric(:,1);
testTargetVector = testNumeric(:,1);
% Gets the data
trainingInputMatrix = trainNumeric(:,2:size(trainNumeric,2));
testInputMatrix = testNumeric(:,2:size(testNumeric,2));
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
net.layers{1,1}.size = 10;
net.layers{2,1}.size = 10;
net.layers{3,1}.size = 10;
net.performFcn = 'mse';
% Connects layers and bias units
net.layerConnect = [0 0 0 0; 1 0 0 0; 0 1 0 0;0 0 1 0];
net.biasConnect = [1;1;1;1];
net.outputConnect = [0 0 0 1];
% Shape the network to the data
%net = configure(net,trainingInputMatrix,trainTargetVector);
net.performParam.regularization = .5;
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
performanceStruct = struct('Net',0);
performanceMatrix = zeros(25,1);
for networkCount = 1 : 30
net = init(net);
[net1,record] = train(net,trainingInputMatrix,trainTargetVector,'useGPU','yes');
trainPerformance = perform(net, trainingInputMatrix,trainTargetVector);
% Calculating percent correct
y = net1(trainingInputMatrix);
% Changes the probability boundary
y = y > .5;
findCorrect = find(y == trainTargetVector);
percentCorrect = size(findCorrect,2) / size(trainTargetVector,2); 
trialPercentPerformance(networkCount) = percentCorrect;
trialTrainPerformance(networkCount) = trainPerformance;
performanceStruct(networkCount).Net = net1;
performanceMatrix(networkCount) = trainPerformance;
end
[~,bestPerformanceIndex] = max(trialPercentPerformance);
net = performanceStruct(bestPerformanceIndex).Net;
% Calculate the average performance
percentCorrect = sum(trialPercentPerformance,1) / size(trialPercentPerformance,1);
trainPerformance = sum(trialPercentPerformance,1) / size(trialPercentPerformance,1);
%plotconfusion(trainTargetVector,y,'Multilayer');
set(findobj(gca,'type','text'),'fontsize',30)
% Finds where the network has failed
error = find(~y == trainTargetVector);
%[xCoordinate,yCoordinate,threshhold,AUC] = perfcurve(trainTargetVector,y,1);
%{
coords = struct('x',0,'y',0,'AUC',0);

coords.x = xCoordinate;
coords.y = yCoordinate;
coords.AUC = AUC;
%save coords;
%}
%{
load coords
hold on
title('ROC Performance','FontSize',30);
xlabel('False Positive Rate','FontSize',30);
ylabel('True Positive Rate','FontSize',30);

plot(xCoordinate,yCoordinate);
plot(coords.x,coords.y);
legend('Ensemble Network', 'Multilayer Network');
%}
