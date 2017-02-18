% Reads Excel File
trainingfileName = 'breastCancerFullDataSet';
[trainNumeric,text,excel] = xlsread(trainingfileName);
net = selforgmap([8 8]);
net = train(net,trainNumeric);
plotsomhits(net,trainNumeric);