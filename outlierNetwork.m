% Reads Excel File
trainingfileName = 'scienceFairBreastCancerData.xls';
[trainNumeric,text,excel] = xlsread(trainingfileName);
[output,target] = linearNetwork;
error = find(~output == target); 