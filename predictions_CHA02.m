function mu = predictions_CHA02(betas_DNN,Pmax,NoOfSetup,modelName,cluster_size)
    K = size(betas_DNN,1);
    L = size(betas_DNN,2);
    nbrOfSetups = size(betas_DNN,3);

    mu = zeros(K+1,L,nbrOfSetups);

    for l = 1:cluster_size:L

        if l == 16
            l = 14;
        end
        if strcmpi(modelName,"MR_PF")
            model = load(".\Model\CANN_MSE\MR_PF_CANN\MR_PF_CANN_"+l+".mat");
        % elseif strcmpi(modelName,"RZF_PF")
        %     model = load(".\Model\HA02_MSE\RZF_PF_ANN\RZF_PF_ANN_"+index+".mat");
        % elseif strcmpi(modelName,"MR_sumSE")
        %     model = load(".\Model\HA02_MSE\MR_sumSE_ANN\MR_sumSE_ANN_"+index+".mat");
        % else
        %     model = load(".\Model\HA02_MSE\RZF_sumSE_ANN\RZF_sumSE_ANN_"+index+".mat");
        end
        betas = zeros(NoOfSetup,cluster_size*K);
        for c = 1:cluster_size
            betas(:,(c-1)*K+1:c*K) = reshape(betas_DNN(:,l+c-1,:),K,[]).';

        end
        betas = 10*log10(betas*1000); %dB scale

        betas = robustScaler(betas,0,1);
        
        betas = betas.';

        betas = reshape(betas,cluster_size*K,1,1,[]);

        result = transformer.model(betas,model.parameters);

        DNNoutput = reshape(result,K*cluster_size + cluster_size,[]);

        for c = 1:cluster_size
            index = [(c-1)*K+1:c*K cluster_size*K + c];
            mu(:,l+c-1,:) = DNNoutput(index,:);
        end
    end
end