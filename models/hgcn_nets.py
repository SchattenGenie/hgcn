import manifolds
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
import layers.hyp_layers as hyp_layers
from layers.layers import FermiDiracDecoder
from utils.math_utils import artanh, tanh
from torch.utils.checkpoint import checkpoint


def get_dim_act_curv(input_dim, hidden_dim, num_layers=5, act='tanh'):
    """
    Helper function to get dimension and activation at every layer.
    :return:
    """
    if not act:
        act = lambda x: x
    else:
        act = getattr(F, act)
    acts = [act] * (num_layers - 1)
    dims = [input_dim] + ([hidden_dim] * (num_layers - 1))
    dims += [hidden_dim]
    acts += [act]
    n_curvatures = num_layers
    curvatures = []
    for _ in range(n_curvatures):
        curvatures.append(nn.Parameter(torch.Tensor([1.])))
    return dims, acts, curvatures


class HGCN(nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self,
                 manifold,
                 c=1.,
                 input_dim=10,
                 hidden_dim=16,
                 edge_dim=0,
                 num_layers=2,
                 dropout=0.05,
                 use_bias=True,
                 use_att=True,
                 agg_direction='in',
                 **kwargs
                ):
        super(HGCN, self).__init__()
        self.manifold = manifold
        assert num_layers > 1
        c = torch.tensor([c]).float()
        dims, acts, curvatures = get_dim_act_curv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        curvatures.append(c)
        self.curvatures = []
        for i, curv in enumerate(curvatures):
            curv = nn.Parameter(curv)
            self.register_parameter("c_{}".format(i), curv)
            self.curvatures.append(manifolds.Curvature(curv))
        # https://github.com/HazyResearch/hgcn/blob/25a3701f8dfbab2341bc18091fedcb0e9bf61395/models/base_models.py#L27
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                        manifold=self.manifold,
                        in_features=in_dim,
                        out_features=out_dim,
                        edge_dim=edge_dim,
                        c_in=c_in,
                        c_out=c_out,
                        dropout=dropout,
                        act=act,
                        use_bias=use_bias,
                        use_att=use_att,
                        agg_direction=agg_direction
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def forward(self, graph):
        x = graph.x
        adj = graph.edge_index
        if hasattr(graph, 'edge_attr'):
            edge_attr = graph.edge_attr
            edge_attr = self.manifold.proj(
                    self.manifold.expmap0(self.manifold.proj_tan0(edge_attr, self.curvatures[0]), c=self.curvatures[0]),
                    c=self.curvatures[0]
            )
        else:
            edge_attr = None
        # initial transformation to manifold
        x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
                c=self.curvatures[0]
        )
        if self.encode_graph:
            for i, layer in enumerate(self.layers):
                # x_hyp = layer.forward(x=x_hyp, adj=adj, edge_attr=edge_attr)
                x_hyp = checkpoint(layer, x_hyp, adj, edge_attr)
                edge_attr = self.manifold.proj(
                    self.manifold.expmap0(self.manifold.logmap0(edge_attr, self.curvatures[i]), self.curvatures[i + 1]),
                    c=self.curvatures[i + 1]
                )
        else:
            x_hyp = self.layers.forward(x)
        return x_hyp


class HGCNResidual(nn.Module):
    """
    Hyperbolic Residual GCN.
    """

    def __init__(self,
                 manifold,
                 c=1.,
                 input_dim=10,
                 hidden_dim=16,
                 edge_dim=0,
                 num_layers=2,
                 dropout=0.05,
                 use_bias=True,
                 use_att=True,
                 agg_direction='in',
                 **kwargs
                ):
        super(HGCNResidual, self).__init__()
        self.manifold = manifold
        assert num_layers > 1
        c = torch.tensor([c]).float()
        dims, acts, curvatures = get_dim_act_curv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        curvatures.append(c)
        self.curvatures = []
        for i, curv in enumerate(curvatures):
            curv = nn.Parameter(curv)
            self.register_parameter("c_{}".format(i), curv)
            self.curvatures.append(manifolds.Curvature(curv))
        # https://github.com/HazyResearch/hgcn/blob/25a3701f8dfbab2341bc18091fedcb0e9bf61395/models/base_models.py#L27
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                        manifold=self.manifold,
                        in_features=in_dim,
                        out_features=out_dim,
                        c_in=c_in,
                        c_out=c_out,
                        edge_dim=edge_dim,
                        dropout=dropout,
                        act=act,
                        use_bias=use_bias,
                        use_att=use_att,
                        agg_direction=agg_direction
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def forward(self, graph):
        x = graph.x
        adj = graph.edge_index
        if hasattr(graph, 'edge_attr'):
            edge_attr = graph.edge_attr
            edge_attr = self.manifold.proj(
                    self.manifold.expmap0(self.manifold.proj_tan0(edge_attr, self.curvatures[0]), c=self.curvatures[0]),
                    c=self.curvatures[0]
            )
        else:
            edge_attr = None
        # initial transformation to manifold
        x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
                c=self.curvatures[0]
        )
        for i, layer in enumerate(self.layers):
            # x_new = layer.forward(x=x_hyp, adj=adj, edge_attr=edge_attr)
            x_new = checkpoint(layer, x_hyp, adj, edge_attr)
            edge_attr = self.manifold.proj(
                self.manifold.expmap0(self.manifold.logmap0(edge_attr, self.curvatures[i]), self.curvatures[i + 1]),
                c=self.curvatures[i + 1]
            )
            if i >= 1:
                x_hyp_new_curv = self.manifold.proj(
                    self.manifold.expmap0(self.manifold.logmap0(x_hyp, self.curvatures[i]), self.curvatures[i + 1]),
                    c=self.curvatures[i + 1]
                )
                x_hyp = self.manifold.proj(
                    self.manifold.mid_point_poincare(x=x_hyp_new_curv, y=x_new, c=self.curvatures[i + 1], manifold=self.manifold),
                    c=self.curvatures[i + 1]
                )
            else:
                x_hyp = x_new
        return x_hyp


class HGCNResidualEmulsionConv(nn.Module):
    """
    Hyperbolic Residual GCN.
    """

    def __init__(self,
                 manifold,
                 c=1.,
                 input_dim=10,
                 hidden_dim=16,
                 edge_dim=0,
                 num_layers=5,
                 dropout=0.05,
                 use_bias=True,
                 use_att=True,
                 agg_direction='in',
                 **kwargs
                ):
        super(HGCNResidualEmulsionConv, self).__init__()
        self.manifold = manifold
        assert num_layers > 1
        c = torch.tensor([c]).float()
        dims, acts, curvatures = get_dim_act_curv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        curvatures.append(c)
        self.curvatures = []
        for i, curv in enumerate(curvatures):
            curv = nn.Parameter(curv)
            self.register_parameter("c_{}".format(i), curv)
            self.curvatures.append(manifolds.Curvature(curv))
        # https://github.com/HazyResearch/hgcn/blob/25a3701f8dfbab2341bc18091fedcb0e9bf61395/models/base_models.py#L27
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicEmulsionConvolution(
                        manifold=self.manifold,
                        in_features=in_dim,
                        out_features=out_dim,
                        edge_dim=edge_dim,
                        c_in=c_in,
                        c_out=c_out,
                        dropout=dropout,
                        act=act,
                        use_bias=use_bias,
                        use_att=use_att,
                        agg_direction=agg_direction
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def forward(self, graph):
        x = graph.x
        adj = graph.edge_index
        orders = graph.orders
        if hasattr(graph, 'edge_attr'):
            edge_attr = graph.edge_attr
        else:
            edge_attr = None
        # initial transformation to manifgold
        x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
                c=self.curvatures[0]
        )

        for i, layer in enumerate(self.layers):
            x_new = layer.forward(x=x_hyp, adj=adj, edge_attr=edge_attr, orders=orders)
            if i >= 1:
                x_hyp_new_curv = self.manifold.proj(
                    self.manifold.expmap0(self.manifold.logmap0(x_hyp, self.curvatures[i]), self.curvatures[i + 1]),
                    c=self.curvatures[i + 1]
                )
                x_hyp = self.manifold.proj(
                    self.manifold.mid_point_poincare(x=x_hyp_new_curv, y=x_new, c=self.curvatures[i + 1], manifold=self.manifold),
                    c=self.curvatures[i + 1]
                )
            else:
                x_hyp = x_new
        return x_hyp