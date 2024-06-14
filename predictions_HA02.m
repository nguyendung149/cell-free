function mu = predictions_HA02(betas_DNN,Pmax,NoOfSetups,modelname)
   K = size(betas_DNN,1);
   L = size(betas_DNN,2);
   nbrOfSetups = size(betas_DNN,3);
   mu = zeros(K+1,nbrOfSetups);
   result = zeros(K+1,1,1,nbrOfSetups);
   for l = 1:L
        betas = reshape(betas_DNN(:,l,:),K,[]).' * 1000;

        v = 0.6;
        betas = sqrt(Pmax) * ((betas.^v)./reshape(sum(betas.^v,2),NoOfSetups,1));

        betas = 10*log10(betas);

        % betas = normalize(betas,"scale","iqr","center","median");
        betas = robustScaler(betas,0,1);

        betas = betas.';
        betas = reshape(betas,K,1,1,[]);

        result = transformer.model(betas,modelname);

        result = reshape(result,K,[]);

        mu(:,l,:) = result;


    end




end