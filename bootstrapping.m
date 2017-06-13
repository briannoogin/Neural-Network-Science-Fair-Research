% Reads Excel File
trainingfileName = 'breastKaggleFixed';
[trainNumeric,text,excel] = xlsread(trainingfileName);
% Gets the data targets
trainTargetVector = trainNumeric(:,1);
% Gets the data
trainingInputMatrix = trainNumeric(:,2:size(trainNumeric,2));
trainingInputMatrix = trainingInputMatrix.';
% Creates network with three layers
net = feedforwardnet(10);
net.numlayers = 4;
% Tranpose data so it works with the toolbox
trainTargetVector = trainTargetVector.';
% Name the layers
net.layers{1}.name = 'Hidden Layer 1';
net.layers{2}.name = 'Hidden Layer 2';
net.layers{3}.name = 'Hidden Layer 3';
net.layers{4}.name = 'Output';
% Designate the activation functions and number of neural units
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'softmax';
net.layers{1,1}.size = 100;
net.layers{2,1}.size = 50;
net.layers{3,1}.size = 25;
% Connects layers and bias units
net.layerConnect = [0 0 0 0; 1 0 0 0; 0 1 0 0;0 0 1 0];
net.biasConnect = [1;1;1;1];
net.outputConnect = [0 0 0 1];
% Shape the network to the data
%net = configure(net,trainingInputMatrix,trainTargetVector);
net.performParam.regularization = .8;
net.trainFcn = 'trainscg';
net.initFcn = 'initlay';
net.layers{1}.netInputFcn = 'netsum';
net.layers{2}.netInputFcn = 'netsum';
net.layers{3}.netInputFcn = 'netsum';
net.performFcn = 'mse';
net.trainParam.max_fail = 2;
% Change train settings and train network 
net.trainParam.showWindow = 0;
trialPercentPerformance = zeros(10,1);
trialTrainPerformance = zeros(10,1);
% Train multiple networks for percent correct
performanceStruct = struct('Net',0,'Prediction',0,'OverallPrediction',0);
trials = 10;
networkSize = 10000;
performanceMatrix = zeros(1,trials);
predictionMatrix = zeros(networkSize,size(trainTargetVector,2));
% Performs trials to determine the best ensemble performance.
for trial = 1 : trials
 for networkCount = 1 : networkSize
 net = init(net);
 bootStrap = zeros(networkSize,size(trainNumeric,2));
 bootTarget = zeros(networkSize,1);
 % Picks a random training sample
 randIndex = randi(size(trainNumeric,1));
 bootStrap(index,:) = trainNumeric(randIndex,:);
 bootTarget(index,:) = trainTargetVector(randIndex,1);
 [trainedNetwork,record] = train(net,bootStrap,bootTarget,'useGPU','yes');
 trainPerformance = perform(net,trainingInputMatrix,trainTargetVector);
 % Prediction based on trained network
 y = trainedNetwork(trainingInputMatrix);
 % Changes the threshold
 y = y > .5;
 performanceStruct(networkCount).Net = trainedNetwork;
 performanceStruct(networkCount).Prediction = y;
 predictionMatrix(networkCount,:) = y;
 end
% Loops through all the predictions and choose the prediction based on
% what most models predicted
threshold = size(y,2) / 2 + 1;
for predictionIndex = 1 : size(y,2)
    if size(find(predictionMatrix(:,predictionIndex)),2) >= threshold;
        y(predictionIndex) = 1; 
    end
end
% Calculate performance
findCorrect = find(y == trainTargetVector);
percentCorrect = size(findCorrect,2) / size(trainTargetVector,2);
performanceStruct(trial).OverallPrediction = y;
performanceMatrix(trial) = percentCorrect;  
 end
end
% Finds where the network has failed
[maxPerformance,maxIndex] = max(performanceMatrix);
y = performanceStruct(maxIndex).OverallPrediction;
error = find(~y == trainTargetVector);
plotconfusion(trainTargetVector,y,'Ensemble');
