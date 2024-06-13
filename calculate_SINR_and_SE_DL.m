function SE_MR_equal = calculate_SINR_and_SE_DL(signal,interference,prelogfactor,gammaEqual,Pmax)
    L = size(signal,1);
    K = size(signal,2);

    SE_MR_equal = zeros(K,1);

    % Scale the square roots of power coefficients to satisfy all the per-AP power constraints
    % These coefficients correspond to the vectors\vect{\mu}_k in (8) in the paper
    normGammaEqual = [];
    for t = 1:size(gammaEqual,2)
        a = norm(gammaEqual(:,t),"fro");
        normGammaEqual = [normGammaEqual a];
    end
    gammaEqual = gammaEqual*sqrt(Pmax)/max(normGammaEqual);

    % Compute the SEs as in (6) in the paper
    
    for k = 1:K
        SINRnumerator = (reshape(signal(:,k),1,L)*reshape(gammaEqual(k,:),L,1)).^2;
        SINRdenominator = 1 - SINRnumerator;

        for i = 1:K
            SINRdenominator = SINRdenominator + (reshape(gammaEqual(i,:),1,L)*(interference(:,:,k,i)*reshape(gammaEqual(i,:),L,1)));
        
        end
        SE_MR_equal(k) = prelogfactor*log2(1 + SINRnumerator/SINRdenominator);

    end
end