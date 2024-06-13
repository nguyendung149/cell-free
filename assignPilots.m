function assigned = assignPilots(K,tau_p,betas)
    assigned = -1 * ones(K,1);
    assigned(1:tau_p) = randperm(tau_p);

    for k = tau_p + 1:K
        [a,l] = max(betas(k,:));
        interference = zeros(tau_p,1);

        for tau = 1:tau_p
            interference(tau) = sum(betas(assigned == tau,l));
        end
        [a,assigned(k)] = min(interference);
    end
end