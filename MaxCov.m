function [X,I_out] = MaxCov(F4,N,I_tr,y)
    X=[];
    size_F4=size(F4);
    if N>size_F4(2)
        N=floor(size_F4(2));
        'N bzw. dl zu gross'
    end
    I_out=[];
    if I_tr==0
        
            corr=[];
            for i=1:size(F4,2)
                if sum(F4(:,i))>1e-6
                    xn=F4(:,i)/sum(F4(:,i));
                    C=cov(xn,y);
                    corr=[corr [abs(C(2));sign(C(2))]];
                else
                    corr=[corr [0;0]];
                end
            end           
            sorted_corr=sort(corr(1,:));
            for i=0:N-1
               [~, I]=max(sorted_corr(end-i)==corr(1,:));
                I_out=[I_out [I;corr(2,I)]];
            end
        for i=1:N
            F=0;
            F=F+F4(:,I_out(1,i));
            X=[X F];
        end
    else
        for i=1:N
            F=0;
            F=F+F4(:,I_tr(1,i));
            X=[X F];
        end  
        I_out=I_tr;
    end
NumberOfFeatures=size(X,2)
end

