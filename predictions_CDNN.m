function [mu,stop_time] = predictions_CDNN(betas_DNN,Pmax,NoOfSetup,modelname,cluster_size)
    K = size(betas_DNN,1);
    L = size(betas_DNN,2);
    nbrOfSetups = size(betas_DNN,3);

    mu = zeros(K+1,L,nbrOfSetups);

    for l = 1:cluster_size:L
        model = modelname((l - 1)/cluster_size + 1);
        if l == 16
            l = 14;
        end
        tic
        betas = zeros(NoOfSetup,cluster_size*K);
        for c = 1:cluster_size
            betas(:,(c-1)*K+1:c*K) = reshape(betas_DNN(:,l+c-1,:),K,[]).';

        end
        betas = 10*log10(betas*1000); %dB scale

        betas = robustScaler(betas,0,1);
        DNNoutput = model.predict(betas).';
        for c = 1:cluster_size
            index = [(c-1)*K+1:c*K cluster_size*K + c];
            mu(:,l+c-1,:) = DNNoutput(index,:);
        end
        stop_time = toc;
    end
end