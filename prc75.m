function pv75  = prc75(data)
    N = length(data);
    N75 = 0.75*(N-1);
    if N75 == fix(N75)
        pv75 = data(N75+1);
    else
        Upperbound = ceil(N75);
        Lowerbound = floor(N75);
        
        UpperValue = data(Upperbound + 1);
        LowerValue = data(Lowerbound + 1);

        p = polyfit([Upperbound Lowerbound],[UpperValue LowerValue],1);

        pv75 = p(1)*N75 + p(2);
        
    end

end