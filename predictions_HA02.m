function [mu,stop_time] = predictions_HA02(betas_DNN,Pmax,NoOfSetups,modelName,isNormalize)
   K = size(betas_DNN,1);
   L = size(betas_DNN,2);
   nbrOfSetups = size(betas_DNN,3);

   mu = zeros(K+1,L,nbrOfSetups);
   result = zeros(K+1,1,1,nbrOfSetups);
   for l = 1:L
        if strcmpi(modelName,"MR_PF")
            model = load(".\Model\HA02_MSE\MR_PF_ANN\MR_PF_ANN_"+l+".mat");
        elseif strcmpi(modelName,"RZF_PF")
            model = load(".\Model\HA02_MSE\RZF_PF_ANN\RZF_PF_ANN_"+l+".mat");
        elseif strcmpi(modelName,"MR_sumSE")
            model = load(".\Model\HA02_MSE\MR_sumSE_ANN\MR_sumSE_ANN_"+l+".mat");
        else
            model = load(".\Model\HA02_MSE\RZF_sumSE_ANN\RZF_sumSE_ANN_"+l+".mat");
        end
        tic;
        betas = reshape(betas_DNN(:,l,:),K,[]).' * 1000;

        v = 0.6;
        betas = sqrt(Pmax) * ((betas.^v)./reshape(sum(betas.^v,2),NoOfSetups,1));

        betas = 10*log10(betas);

        % betas = normalize(betas,"scale","iqr","center","median");
        if isNormalize
            betas = robustScaler(betas,0,1);
        end
        betas = betas.';
        betas = reshape(betas,K,1,1,[]);

        result = transformer.model(betas,model.parameters);

        result = reshape(result,K+1,[]);

        mu(:,l,:) = result;

        stop_time = toc;


    end

end