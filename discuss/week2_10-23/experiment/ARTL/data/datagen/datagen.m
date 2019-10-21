function datagen()
% generate the 216 cross-domain classification tasks

load('comp');
load('rec');
load('sci');
load('talk');
gen_dataset('comp','rec');
gen_dataset('comp','sci');
gen_dataset('comp','talk');
gen_dataset('rec','sci');
gen_dataset('rec','talk');
gen_dataset('sci','talk');
fprintf('the 216 cross-domain tasks are generated successfully!!!\n');
end

function gen_dataset(c1,c2)
% generate 36 tasks from top-categories c1 and c2

fprintf('generate the 36 tasks in group %s_vs_%s...\n',c1,c2);
iTask = 1;
domains = [1,2,3,4];
for i = 1:4
    for j = i+1:4
        load(strcat(c1,'.mat'));
        X_src_i = eval(strcat('X',num2str(i)));
        Y_src_i = ones(size(X_src_i,2),1);
        X_src_j = eval(strcat('X',num2str(j)));
        Y_src_j = ones(size(X_src_j,2),1);
        ij = setdiff(domains,[i,j]);
        ii = ij(1);
        X_tar_i = eval(strcat('X',num2str(ii)));
        Y_tar_i = ones(size(X_tar_i,2),1);
        jj = ij(2);
        X_tar_j = eval(strcat('X',num2str(jj)));
        Y_tar_j = ones(size(X_tar_j,2),1);
        
        for k = 1:4
            for l = k+1:4
                load(strcat(c2,'.mat'));
                X_src_k = eval(strcat('X',num2str(k)));
                Y_src_k = -ones(size(X_src_k,2),1);
                X_src_l = eval(strcat('X',num2str(l)));
                Y_src_l = -ones(size(X_src_l,2),1);
                kl = setdiff(domains,[k,l]);
                kk = kl(1);
                X_tar_k = eval(strcat('X',num2str(kk)));
                Y_tar_k = -ones(size(X_tar_k,2),1);
                ll = kl(2);
                X_tar_l = eval(strcat('X',num2str(ll)));
                Y_tar_l = -ones(size(X_tar_l,2),1);
                
                X_src = [X_src_i,X_src_j,X_src_k,X_src_l]; %#ok<*NASGU>
                X_tar = [X_tar_i,X_tar_j,X_tar_k,X_tar_l];
                Y_src = [Y_src_i;Y_src_j;Y_src_k;Y_src_l];
                Y_tar = [Y_tar_i;Y_tar_j;Y_tar_k;Y_tar_l];
                
                save(strcat('data/',c1,'_vs_',c2,'_',num2str(iTask)),'X_src','X_tar','Y_src','Y_tar');
                fprintf('generate dataset %s_vs_%s_%d successfully!\n',c1,c2,iTask);
                iTask = iTask+1;
            end
        end
    end
end
clear all;
end