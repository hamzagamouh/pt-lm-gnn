
def skip_letters(k,pdb_id,chain_id):
    skip_let=False

    if pdb_id=="2UZP" and chain_id=="A" and k==0:
        skip_let=True
    if pdb_id=="1FSU" and chain_id=="A" and k==50:
        skip_let=True
    
    if pdb_id=="3KQA" and chain_id=="B" and k==66:
        skip_let=True

    if pdb_id=="1P49" and chain_id=="A" and k==52:
        skip_let=True
    
    if pdb_id=="3KNI" and chain_id=="Y" and k==100:
        skip_let=True
    
    if pdb_id=="3KNI" and chain_id=="H" and k==163:
        skip_let=True
    
    if pdb_id=="3KNK" and chain_id=="Z" and k==176:
        skip_let=True
    
    if pdb_id=="3OHJ" and chain_id=="X" and k==92:
        skip_let=True

    if pdb_id=="3CAO" and chain_id=="A" and k==1:
        skip_let=True
        
    if pdb_id=="2YHW" and chain_id=="A" and k==170:
        skip_let=True
    
    if pdb_id=="3V2D" and chain_id=="F" and k==202:
        skip_let=True
    
    if pdb_id=="3SWI" and chain_id=="A" and k==66:
        skip_let=True
        
    if pdb_id=="4E6R" and chain_id=="A" and k==0:
        skip_let=True

    return skip_let


def correct_sequence(seq,pdb_id,chain_id):
    if pdb_id=="1UC9":
        l=list(seq)
        l[133:149]=['G']*len(l[133:149])
        seq=''.join(l)

    if pdb_id=="3BFN":
        l=list(seq)
        l[281:]=['G']*len(l[281:])
        seq=''.join(l)

    if pdb_id=="3PIF":
        l=list(seq)
        l[429:448]=['G']*len(l[429:448])
        l[892:907]=['G']*len(l[892:907])
        seq=''.join(l)

    if pdb_id=="2GMK":
        l=list(seq)
        l[0]="E"
        seq=''.join(l)

    if pdb_id=="1H1A" and chain_id=="A":
        l=list(seq)
        l[0]="E"
        seq=''.join(l)

    if pdb_id=="1VCL" and chain_id=="A":
        l=list(seq)
        l[0]="E"
        seq=''.join(l)

    if pdb_id=="1FSU" and chain_id=="A":
        l=list(seq)
        l[49]="T"
        seq=''.join(l)

    if pdb_id=="1HDH" and chain_id=="A":
        l=list(seq)
        l[48]="A"
        seq=''.join(l)

    if pdb_id=="1Q2E" and chain_id=="B":
        l=list(seq)
        l[0]="E"
        seq=''.join(l)

    if pdb_id=="1PL3" and chain_id=="A":
        l=list(seq)
        l[0]="E"
        seq=''.join(l)

    if pdb_id=="1CPO" and chain_id=="A":
        l=list(seq)
        l[0]="E"
        seq=''.join(l)

    if pdb_id=="2FUQ" and chain_id=="A":
        l=list(seq)
        l[0]="E"
        seq=''.join(l)
    return seq