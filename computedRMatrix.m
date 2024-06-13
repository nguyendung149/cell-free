function R = computedRMatrix(M,theta,ASDdeg)
    ASD = ASDdeg*pi/180;
    antennaSpacing = 0.5;
    
    % The correlation matrix has a Toeplitz structure, so we only need to
    % compute the first row of the matrix
    
    firstRow = zeros(M,1);

    for column = 1:M
        % Compute the approximated integral as in (2.24)
        firstRow(column) = exp(1i*2*pi*antennaSpacing*sin(theta)*(column-1))*exp(-ASD^2/2 *(2*pi*antennaSpacing*cos(theta)*(column-1))^2);

    end

    R_temp = toeplitz(firstRow);

    for t = 1:size(R_temp,1)
        R(:,t) = R_temp(t,:);
    end
end