function mu = pred_func_test(betas_DNN,modelname)
    
    K = 20;
    L = 16;
    nbrOfSetups = 1000;

    mu = zeros(K,L,nbrOfSetups);

    for l = 1:L
        
        mu(:,l,:) = predict(modelname(l),reshape(betas_DNN(:,l,:),20,[]).').';




    end



end