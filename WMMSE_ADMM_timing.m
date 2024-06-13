function mu_MR_WMMSE = WMMSE_ADMM_timing(L,K,Pmax,a_MR,B_MR)
% The functions in this script implement Algorithm 2 and Algorithm 1 using ADMM
% This function is for saving run time, its implementataion is the same as the last function in this script.
% The difference from the last function is that arrays except power coefficients are not saved for a fair comparison of run times.    
    % Initialize the square roots of the power coefficients
    mu_MR_WMMSE = 0.1*sqrt(Pmax/K)*ones(L,K);

    % Solution accuracy (epsilon_wmmse)
    delta = 0.01;

    % ADMM penalty parameter
    penalty = 0.001;

    % Initialize the objective function as zero
    objLower = 0;

    % This is for computing the objective function and computing the other terms later
    SINRnumerator = abs(sum(mu_MR_WMMSE.*a_MR)).^2;
    SINRdenominator = ones(K,1)';
    for k = 1:K
        for i = 1:K
            SINRdenominator(k) = SINRdenominator(k) + mu_MR_WMMSE(:,i).'*B_MR(:,:,k,i)*mu_MR_WMMSE(:,i);

        end
    end
    SINR = SINRnumerator./(SINRdenominator - SINRnumerator);

    % Current objective function
    objUpper = sum(log2(1 + SINR));

    % Continue iterations until stopping criterion in (52) is satisfied (prelogfactors are omitted)
    while abs(objUpper - objLower)^2 >delta
        % Update the old objective by the current objective
        objLower = objUpper;

        % Equation (53)
        v = sqrt(SINRnumerator)./SINRdenominator;
        
        % Equation (56)
        e = 1 - SINRnumerator./SINRdenominator;

        % Equation (55)
        w = 1./e;

        % Make preparations for ADMM algorithm in Algorithm 1
        Ainv = zeros(L,L,K);
        c = zeros(L,K);

        for k = 1:K
            A = (penalty/2)*eye(L);
            for i = 1: K
                A = A + w(i) .* v(i)^2 .* B_MR(:,:,i,k);
            end
            Ainv(:,:,k) = A^-1;

            c(:,k) = w(k) .* v(k) .* a_MR(:,k);

        end
        % Dual variable initialization for ADMM
        g = zeros(L,K);

        % Start the ADMM
        % Initial large difference in (51) to start the algoritm
        diff = 100;

        % Set the iteration counter for ADMM
        inner_iteration = 0;

        % Perturbed random initialization mentioned in the paper
        q = mu_MR_WMMSE.*(1 + rand(L,K));

        % Run Algorithm 1 until the stopping criterion in (51) is satisfied
        while diff > 0.001
            inner_iteration = inner_iteration + 1;
            % Update the first block of primal variables as in (48)
            
            for k = 1:K
                c2 = c(:,k) + (penalty/2) * (q(:,k) + g(:,k));
                
                mu_MR_WMMSE(:,k) = Ainv(:,:,k) * c2;


            end

            % Update the second block of primal variables as in (49)

            q = mu_MR_WMMSE - g;
            
            q_norm = [];
            for t = 1:size(q,1)
                a = norm(q(t,:),"fro");
                q_norm = [q_norm a];

            end
            arrayFind = find(q_norm>sqrt(Pmax));
            for l = 1:length(arrayFind)
                q(arrayFind(l),:) = q(arrayFind(l),:) * sqrt(Pmax)/q_norm(arrayFind(l));


            end

            % Update dual variable g as in (50) 
            g = q - mu_MR_WMMSE + g;

            % To prevent any misconvergence issues in the first iterations, we guarentee at least 5 ADMM iterations are run
            if inner_iteration > 5
                diff = norm(mu_MR_WMMSE-q,"fro")/norm(mu_MR_WMMSE,"fro");

        
            end




        end
        % Update the variables and compute the new objective
        SINRnumerator = abs(sum(mu_MR_WMMSE .* a_MR)).^2;
        SINRdenominator = ones(K,1)';
        for k = 1:K
            for i = 1:K
               SINRdenominator(k) = SINRdenominator(k) +  mu_MR_WMMSE(:,i).'* B_MR(:,:,k,i) * mu_MR_WMMSE(:,i);
            end


        end
        SINR = SINRnumerator./(SINRdenominator - SINRnumerator);
        objUpper = sum(log2(1 + SINR));




    end
    % Square roots of the power coefficients
    mu_MR_WMMSE = mu_MR_WMMSE.';
end