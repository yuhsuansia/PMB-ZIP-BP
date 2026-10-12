function [outputDA] = dataAssociationBP(inputDA)
%DATAASSOCIATIONBP Closed-form BP update for one measurement node.

inputDA = inputDA(2,:)./inputDA(1,:);
sumInputDA = 1 + sum(inputDA,2);
outputDA = 1 ./ (repmat(sumInputDA,[1,size(inputDA,2)]) - inputDA);

% Fall back to a hard assignment if numerical overflow creates NaNs.
if(any(isnan(outputDA)))
    outputDA = zeros(size(outputDA));
    [~,index] = max(inputDA(1,:));
    outputDA(index(1)) = 1;
end
end
