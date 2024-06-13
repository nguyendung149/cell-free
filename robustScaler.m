function scaledData = robustScaler(data,with_centering,with_scaling)
   
    scaledData = data;
    [numRows, numCols] = size(data);
    quantiles = [];
    for i = 1: numCols
        columnData = data(:,i);
        columnData = sort(columnData,"ascend");
        quantiles = [quantiles ;prc25(columnData),prc75(columnData)];

    end
    scale = quantiles(:,2) - quantiles(:,1);
    constantMask  = scale < 10 * eps;
    scale(constantMask) = 1;

    center = median(data,1);
    if with_centering
        scaledData = scaledData - center;
    
    end
    if with_scaling
        for i = 1:size(scaledData,2)
            scaledData(:,i) = scaledData(:,i) / scale(i);
        end
        
    end 
end