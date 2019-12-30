"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.utils import scatter_
from layers.att_layers import DenseAtt
from utils.math_utils import artanh, tanh
import numpy as np


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, edge_dim, dropout, act, use_bias, use_att, agg_direction):
        super(HyperbolicGraphConvolution, self).__init__()
        self.manifold = manifold
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, use_att, out_features, dropout, agg_direction, edge_dim=edge_dim)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.c_out = c_out

    def forward(self, x, adj, edge_attr, **kwargs):
        # x, adj, *_ = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj, edge_attr)
        h = self.hyp_act.forward(h)
        return h
    

def extract_subgraph(h, adj, edge_attr, order):
    adj_selected = adj[:, order]
    edge_attr_selected = edge_attr[order, :]
    nodes_selected = adj_selected.unique()
    h_selected = h[nodes_selected]
    nodes_selected_new = torch.arange(len(nodes_selected))
    dictionary = dict(zip(nodes_selected.cpu().numpy(), nodes_selected_new.cpu().numpy()))
    adj_selected_new = torch.tensor(np.vectorize(dictionary.get)(adj_selected.cpu().numpy())).long().to(adj)
    return h_selected, nodes_selected, edge_attr_selected, adj_selected_new


def mid_point_poincare(x, y, c, manifold):
    sqrt_c = c ** 0.5
    r = 0.5

    x_y = manifold.mobius_add(-x, y, c=c)
    norm = torch.clamp_min(x_y.norm(dim=-1, keepdim=True, p=2), 1e-15)
    x_y_r = tanh(r * artanh(sqrt_c * norm)) * (x_y / norm) / sqrt_c
    mid_point = manifold.mobius_add(x, x_y_r, c=c)
    return mid_point


class HyperbolicEmulsionConvolution(nn.Module):
    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, agg_direction, edge_dim):
        super(HyperbolicEmulsionConvolution, self).__init__()
        self.manifold = manifold
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, use_att, out_features, dropout, agg_direction, edge_dim=edge_dim)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.c_out = c_out

    def forward(self, x, adj, edge_attr, orders):
        # x, adj, orders, *_ = input
        h = self.linear.forward(x)
        for order in orders:
            # print(order.shape, order.sum())
            if order.sum():
                h_selected, nodes_selected, edge_attr_selected, adj_selected_new = extract_subgraph(
                    h,
                    adj,
                    edge_attr,
                    order=order
                )
                """
                h[nodes_selected] = self.manifold.proj(
                    self.manifold.mobius_add(
                        h[nodes_selected],
                        self.agg.forward(h[nodes_selected], adj_selected_new, edge_attr_selected),
                        c=self.c_out
                    ),
                    c=self.c_out
                )
                """
                h[nodes_selected] = self.manifold.proj(
                    mid_point_poincare(
                        h[nodes_selected],
                        self.agg.forward(h[nodes_selected], adj_selected_new, edge_attr_selected),
                        c=self.c_out,
                        manifold=self.manifold
                    ),
                    c=self.c_out
                )
                # print(h)
                # TODO: average?
                # h = h + self.agg.forward(h, adj[:, order])
        h = self.hyp_act.forward(h)
        # output = h, adj, orders
        return h


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias: 
            bias = self.manifold.proj_tan0(self.bias, self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res
        

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
                self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """
    # TODO: bi directional
    def __init__(self, manifold, c, use_att, in_features, dropout, agg_direction='in', edge_dim=0):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.use_att = use_att
        if agg_direction == 'in':
            self.i = 0
        elif agg_direction == 'out':
            self.i = 1
        else:
            RaiseError("Nononono")
        self.in_features = in_features
        self.dropout = dropout
        if use_att:
            print(edge_dim)
            self.att = DenseAtt(2 * in_features + edge_dim, dropout, lambda x: x)

    def forward(self, x, adj, edge_attr):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            # TODO : merge in sparse att layer
            adj_att = self.att(x_tangent, adj, edge_attr)
        support_t = (
                x_tangent +
                scatter_('add', x_tangent[adj[self.i]] * adj_att, adj[self.i], dim=0, dim_size=len(x)) +
                scatter_('mean', edge_attr, adj[self.i], dim=0, dim_size=len(x)) +
                scatter_('mean', edge_attr, adj[1 - self.i], dim=0, dim_size=len(x))
                    )
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}, use_att={}'.format(
                self.c, self.use_att
        )


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
                self.c_in, self.c_out
        )
