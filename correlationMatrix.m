data = xlsread('30featuresbreast.xlsx');
data2 = xlsread('correlationMatrixdata10features.xlsx');
dataNoOutputVariables = data(:,2:size(data,2));
corMatrix = corrcoef(data);
corMatriz2 = corrcoef(dataNoOutputVariables);
corMatrix3 =  corrcoef(data2);
