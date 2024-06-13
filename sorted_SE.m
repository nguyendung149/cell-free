function sorted_SE = sorted_SE(SE)
    K = size(SE,1);
    nbrOfSetups = size(SE,2);

    A = reshape(SE(:,1:nbrOfSetups),K*nbrOfSetups,1);
    
    [B,index] = sort(A(:,1));
    sorted_SE = A(index);

end