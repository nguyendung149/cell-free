function scaledData = robustScaler_T(data)  
meanData = mean(data)*0;
stdDevData = std(data);

% Perform Z-Score normalization on the data by subtracting the mean and dividing by the standard deviation
scaledData = (data - meanData)./stdDevData;

end