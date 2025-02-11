import torch


def PCC(m):
  """Compute the Pearson’s correlation coefficients."""
  fact = 1.0 / (m.size(1) - 1)
  m = m - torch.mean(m, dim=1, keepdim=True)
  mt = m.t() 
  c = fact * m.matmul(mt).squeeze()    
  d = torch.diag(c, 0)
  std = torch.sqrt(d)
  c /= std[:, None]
  c /= std[None, :]
  return c


class EGA(torch.nn.Module):
    """Embedding Graph Alignment Loss"""
    def __init__(self):
        super(EGA, self).__init__()
        self.node_weight = 1
        self.edge_weight = 0.3

    def forward(self, ft, fs):
        X = torch.cat((ft, fs), 0)
        C = PCC(X) 
        n = C.shape[0]//2
    
        Et = C[0:n, 0:n] # compute teacher edge matrix
        Es = C[n:, n:] # compute student edge matrix
        Nts= C[0:n, n:] # compute node matrix
      
        loss_edge = torch.norm((Et-Es), 2)  
        loss_node = torch.norm((Nts - torch.eye(Nts.shape[0]).cuda()), 2)
        loss = self.node_weight * loss_node + self.edge_weight * loss_edge 
        return loss