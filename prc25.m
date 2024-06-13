function pv25  = prc25(data)
    N = length(data);
    N25 = 0.25*(N-1);
    if N25 == fix(N25)
        pv25 = data(N25+1);
    else
        Upperbound = ceil(N25);
        Lowerbound = floor(N25);
        
        UpperValue = data(Upperbound + 1);
        LowerValue = data(Lowerbound + 1);

        p = polyfit([Upperbound Lowerbound],[UpperValue LowerValue],1);

        pv25 = p(1)*N25 + p(2);
        
    end

end