
function [X I_out] = FeatureExtractionTEST(filename,interval,y,I_tr,N,dl)
    if y>0
        X=csvread(strcat(num2str(dl),'train',num2str(length(y))));
    elseif y+strcmp(filename,'test_')==0
        X=csvread(strcat(num2str(dl),'val',num2str(interval(2)-interval(1)+1)));
    else
        X=csvread(strcat(num2str(dl),'test',num2str(interval(2)-interval(1)+1)));
    end
    F4=X(:,2:end);
    MEAN=X(:,1);
    [X,I_out]=MaxCov(F4,N,I_tr,y);
    X=[X MEAN];
end
