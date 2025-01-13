#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import random
import warnings
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Literal

import einops
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tabpfn.model.encoders import (
    InputNormalizationEncoderStep,
    LinearInputEncoderStep,
    MulticlassClassificationTargetEncoder,
    NanHandlingEncoderStep,
    RemoveDuplicateFeaturesEncoderStep,
    RemoveEmptyFeaturesEncoderStep,
    SequentialEncoder,
    VariableNumFeaturesEncoderStep,
)
from tabpfn.model.encoders import (
    LinearInputEncoderStep,
    NanHandlingEncoderStep,
    SequentialEncoder,
)
from tabpfn.model.layer import PerFeatureEncoderLayer

DEFAULT_EMSIZE = 128


class LayerStack(nn.Module):
    """Similar to nn.Sequential, but with support for passing keyword arguments
    to layers and stacks the same layer multiple times.
    """

    def __init__(
        self,
        *,
        layer_creator: Callable[[], nn.Module],
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        half_layers: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if half_layers:
            assert (
                self.min_num_layers_layer_dropout == self.num_layers
            ), "half_layers only works without layer dropout"
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)  # type: ignore
            else:
                x = layer(x, **kwargs)

        return x


class PerFeatureTransformer(nn.Module):
    """A Transformer model processes a token per feature and sample.

    This model extends the standard Transformer architecture to operate on a
    per-feature basis.
    It allows for processing each feature separately while still leveraging the
    power of self-attention.

    The model consists of an encoder, decoder, and optional components such
    as a feature positional embedding and a separate decoder for each feature.
    """

    # TODO: Feel like this could be simplified a lot from this part downwards
    def __init__(  # noqa: C901, D417, PLR0913
        self,
        *,
        decoder_n_out: int = 1,
        encoder: nn.Module | None = None,
        ninp: int = DEFAULT_EMSIZE,
        nhead: int = 4,
        nhid: int = DEFAULT_EMSIZE * 4,
        nlayers: int = 10,
        y_encoder: nn.Module | None = None,
        decoder_dict: dict[str, tuple[type[nn.Module] | None, int]] | None = None,
        init_method: str | None = None,
        activation: Literal["gelu", "relu"] = "gelu",
        recompute_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
        repeat_same_layer: bool = False,
        dag_pos_enc_dim: int = 0,
        features_per_group: int = 1,
        feature_positional_embedding: (
            Literal[
                "normal_rand_vec",
                "uni_rand_vec",
                "learned",
                "subspace",
            ]
            | None
        ) = None,
        zero_init: bool = True,
        use_separate_decoder: bool = False,
        nlayers_decoder: int | None = None,
        use_encoder_compression_layer: bool = False,
        precomputed_kv: (
            list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        cache_trainset_representation: bool = False,
        seed: int | None = None,
        # TODO: List explicitly
        **layer_kwargs: Any,
    ):
        """Initializes the PerFeatureTransformer module.

        Args:
            encoder:
                Pass a nn.Module that takes in a batch of sequences of inputs and
                returns something of the shape (seq_len, batch_size, ninp)
            ninp: Input dimension, also called the embedding dimension
            nhead: Number of attention heads
            nhid: Hidden dimension in the MLP layers
            nlayers:
                Number of layers, each consisting of a multi-head attention layer and
                an MLP layer
            y_encoder:
                A nn.Module that takes in a batch of sequences of outputs and
                returns something of the shape (seq_len, batch_size, ninp)
            decoder_dict: Document this (TODO)
            activation: An activation function, "gelu" or "relu"
            recompute_layer:
                If True, the transformer layers will be recomputed on each
                forward pass in training. This is useful to save memory.
            min_num_layers_layer_dropout:
                If this is set, it enables to drop the last
                layers randomly during training up to this number.
            repeat_same_layer:
                If True, the same layer will be used for all layers.
                This is useful to save memory on weights.
            features_per_group:
                If > 1, the features will be grouped into groups of this
                size and the attention is across groups.
            feature_positional_embedding:
                There is a risk that our models confuse
                features with each other. This positional embedding is added to the
                features to help the model distinguish them.
                We recommend setting this to "subspace".
            zero_init:
                If True, the last sublayer of each attention and MLP layer will
                be initialized with zeros.
                Thus, the layers will start out as identity functions.
            seed: The seed to use for the random number generator.
            use_separate_decoder: If True, the decoder will be separate from the encoder
            nlayers_decoder:
                If use_separate_decoder is True, this is the number of
                layers in the decoder. The default is to use 1/3 of the layers for the
                decoder and 2/3 for the encoder.
            use_encoder_compression_layer: Experimental
            precomputed_kv: Experimental
            layer_kwargs:
                TODO: document.
                for now have a look at layer.py:PerFeatureEncoderLayer.
        """

        super().__init__()

        # -- X encoder
        num_features = 1
        inputs_to_merge = {"main": {"dim": num_features}}
        inputs_to_merge["nan_indicators"] = {"dim": num_features}
        encoder_steps = [
            RemoveEmptyFeaturesEncoderStep(),
            NanHandlingEncoderStep(keep_nans=True),
            VariableNumFeaturesEncoderStep(
                num_features=num_features,
                normalize_by_used_features=False,
                in_keys=["nan_indicators"],
                out_keys=["nan_indicators"],
            ),
            InputNormalizationEncoderStep(
                normalize_on_train_only=True,
                normalize_to_ranking=False,
                normalize_x=True,
                remove_outliers=False,
            ),
            VariableNumFeaturesEncoderStep(
                num_features=num_features,
                normalize_by_used_features=True,
            ),
            LinearInputEncoderStep(
                num_features=sum([i["dim"] for i in inputs_to_merge.values()]),
                emsize=DEFAULT_EMSIZE,
                bias=False,
                in_keys=tuple(inputs_to_merge),
                out_keys=("output",),
            ),
        ]
        encoder = SequentialEncoder(*encoder_steps, output_key="output")

        # -- y encoder
        num_inputs = 1
        inputs_to_merge = [{"name": "main", "dim": num_inputs}, {"name": "nan_indicators", "dim": num_inputs}]
        steps = [
            NanHandlingEncoderStep(),
            # if max_num_classes >= 2:
            #     steps += [MulticlassClassificationTargetEncoder()]
            LinearInputEncoderStep(
                num_features=sum([i["dim"] for i in inputs_to_merge]),  # type: ignore
                emsize=DEFAULT_EMSIZE,
                in_keys=tuple(i["name"] for i in inputs_to_merge),  # type: ignore
                out_keys=("output",),
            )
        ]
        y_encoder = SequentialEncoder(*steps, output_key="output")

        # Rest
        layer_creator = lambda: PerFeatureEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            activation=activation,
            zero_init=zero_init,
            **layer_kwargs,
        )
        nlayers_encoder = nlayers

        self.encoder = encoder
        self.y_encoder = y_encoder
        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.init_method = init_method
        self.features_per_group = features_per_group
        self.transformer_decoder = None
        self.global_att_embeddings_for_compression = None
        self.dag_pos_enc_dim = dag_pos_enc_dim
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)  # noqa: S311

        self.transformer_encoder = LayerStack(
            layer_creator=layer_creator,
            num_layers=nlayers_encoder,
            recompute_each_layer=recompute_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )
        self.decoder = nn.Sequential(
                nn.Linear(ninp, nhid),
                nn.GELU(),
                nn.Linear(nhid, decoder_n_out),
        )

        self.feature_positional_embedding = feature_positional_embedding
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        # Device on which the generator was last initialized.
        # If loading from a checkpoint, this might be false,
        # but it will be set to the correct device on the first forward pass.
        self.generator_device = "cpu"
        self._init_rnd()

    def _init_rnd(self) -> None:
        self.generator = SerializableGenerator(device=self.generator_device)
        if self.seed:  # This can be none if set outside of the model.
            self.generator.manual_seed(self.seed)

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """Sets the save_peak_mem_factor for all layers.

        This factor controls how much memory is saved during the forward pass
        in inference mode.

        Setting this factor > 1 will cause the model to save more memory during
        the forward pass in inference mode.

        A value of 8 is good for a 4x larger width in the fully-connected layers.
        and yields a situation were we need around
        `2*num_features*num_items*emsize*2` bytes of memory

        for a forward pass (using mixed precision).

        WARNING: It should only be used with post-norm.

        Args:
            factor: The save_peak_mem_factor to set. Recommended value is 8.
        """
        for layer in self.transformer_encoder.layers:
            assert hasattr(
                layer,
                "save_peak_mem_factor",
            ), "Layer does not have save_peak_mem_factor"
            layer.save_peak_mem_factor = factor  # type: ignore

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("features_per_group", 1)
        state.setdefault("feature_positional_embedding", None)
        super().__setstate__(state)

    # TODO(eddiebergman): Can we just replace this with specific calls
    # such as forward, forward_with_test, forward_with_style?
    # The documentation generator complains about this function because we are
    # documenting parameters that don't exist in the signature
    def forward(self, train_x, train_y, test_x, categorical_inds = None) -> dict[str, torch.Tensor]:  # noqa: D417
        """Performs a forward pass through the model.


        Args:
            train_x: torch.Tensor
                The input data for the training set.
            train_y: torch.Tensor
                The target data for the training set.
            test_x: torch.Tensor
                The input data for the test set.
            categorical_inds: list[int]
                The indices of categorical features.

        Returns:
            The output of the model, which can be a tensor or a dictionary of tensors.
        """
        self._init_rnd()


        x = torch.cat((train_x, test_x), dim=0)
        return self._forward(x, train_y, single_eval_pos=len(train_y), categorical_inds=categorical_inds)


    def _forward(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor | dict,
        y: torch.Tensor | dict,
        *,
        single_eval_pos: int | None = None,
        only_return_standard_out: bool = True,
        style: torch.Tensor | None = None,
        data_dags: list[Any] | None = None,
        categorical_inds: list[int] | None = None,
        half_layers: bool = False,
    ) -> Any | dict[str, torch.Tensor]:
        """The core forward pass of the model.

        Args:
            x: The input data. Shape: `(seq_len, batch_size, num_features)`
            y: The target data. Shape: `(seq_len, batch_size)`
            single_eval_pos:
                The position to evaluate at. If `None`, evaluate at all positions.
            only_return_standard_out: Whether to only return the standard output.
            style: The style vector.
            data_dags: The data DAGs for each example in the batch.
            categorical_inds: The indices of categorical features.

        Returns:
            A dictionary of output tensors.
        """
        assert style is None
        assert y is not None
        assert single_eval_pos

        single_eval_pos_ = single_eval_pos or 0
        x = {"main": x}
        y = {"main": y}
        seq_len, batch_size, num_features = x["main"].shape

        for k in x:
            num_features_ = x[k].shape[2]

            # pad to multiple of features_per_group
            missing_to_next = (
                self.features_per_group - (num_features_ % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )

        # Splits up the input into subgroups
        for k in x:
            x[k] = einops.rearrange(
                x[k],
                "s b (f n) -> b s f n",
                n=self.features_per_group,
            )  # s b f -> b s #groups #features_per_group

        # We have to re-work categoricals based on the subgroup they fall into.
        categorical_inds_to_use: list[list[int]] | None = None
        if categorical_inds is not None:
            new_categorical_inds = []
            n_subgroups = x["main"].shape[2]

            for subgroup in range(n_subgroups):
                subgroup_lower = subgroup * self.features_per_group
                subgroup_upper = (subgroup + 1) * self.features_per_group
                subgroup_indices = [
                    i - subgroup_lower
                    for i in categorical_inds
                    if subgroup_lower <= i < subgroup_upper
                ]
                new_categorical_inds.append(subgroup_indices)

            categorical_inds_to_use = new_categorical_inds

        for k in y:
            if y[k].ndim == 1:
                y[k] = y[k].unsqueeze(-1)
            if y[k].ndim == 2:
                y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

            y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1

            if y[k].shape[1] < x["main"].shape[1]:
                assert (
                    y[k].shape[1] == single_eval_pos_
                    or y[k].shape[1] == x["main"].shape[1]
                )
                assert k != "main" or y[k].shape[1] == single_eval_pos_, (
                    "For main y, y must not be given for target"
                    " time steps (Otherwise the solution is leaked)."
                )
                if y[k].shape[1] == single_eval_pos_:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],
                                x["main"].shape[1] - y[k].shape[1],
                                y[k].shape[2],
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,
                    )

            y[k] = y[k].transpose(0, 1)  # b s 1 -> s b 1

        # making sure no label leakage ever happens
        y["main"][single_eval_pos_:] = torch.nan

        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
        ).transpose(0, 1)

        del y
        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )

        extra_encoders_args = {}
        if categorical_inds_to_use is not None and isinstance(
            self.encoder,
            SequentialEncoder,
        ):
            extra_encoders_args["categorical_inds"] = categorical_inds_to_use

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")

        embedded_x = einops.rearrange(
            self.encoder(
                x,
                single_eval_pos=single_eval_pos_,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )  # b s f 1 -> b s f e
        del x

        embedded_x, embedded_y = self.add_embeddings(
            embedded_x,
            embedded_y,
            data_dags=data_dags,
            num_features=num_features,
            seq_len=seq_len,
        )
        del data_dags

        # b s f e + b s 1 e -> b s f+1 e
        embedded_input = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)

        if torch.isnan(embedded_input).any():
            raise ValueError(
                f"There should be no NaNs in the encoded x and y."
                "Check that you do not feed NaNs or use a NaN-handling enocder."
                "Your embedded x and y returned the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_y).any()=}",
            )
        del embedded_y, embedded_x

        encoder_out = self.transformer_encoder(
            (
                embedded_input
                if not self.transformer_decoder
                else embedded_input[:, :single_eval_pos_]
            ),
            single_eval_pos=single_eval_pos,
        )  # b s f+1 e -> b s f+1 e

        # # If we are using a decoder
        # if self.transformer_decoder:
        #     assert not half_layers
        #     assert encoder_out.shape[1] == single_eval_pos_
        #
        #     if self.global_att_embeddings_for_compression is not None:
        #         # TODO: fixed number of compression tokens
        #         train_encoder_out = self.encoder_compression_layer(
        #             self.global_att_embeddings_for_compression,
        #             att_src=encoder_out[:, single_eval_pos_],
        #             single_eval_pos=single_eval_pos_,
        #         )
        #
        #     test_encoder_out = self.transformer_decoder(
        #         embedded_input[:, single_eval_pos_:],
        #         single_eval_pos=0,
        #         att_src=encoder_out,
        #     )
        #     encoder_out = torch.cat([encoder_out, test_encoder_out], 1)
        #     del test_encoder_out

        del embedded_input

        # take last embedding of last row and map to result
        # out: s b e
        test_encoder_out = encoder_out[:, -1:, -1]
        # test_encoder_out = einops.rearrange(test_encoder_out, "b s e -> b (s e)")
        output_decoded = self.decoder(test_encoder_out)

        return output_decoded

    def add_embeddings(  # noqa: C901, PLR0912
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        data_dags: Iterable[nx.DiGraph] | None,
        num_features: int,
        seq_len: int,
        cache_embeddings: bool = False,
        use_cached_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.generator_device != self.generator.device
            or self.generator_device != x.device
        ):
            self.generator_device = x.device
            self._init_rnd()

        if self.feature_positional_embedding == "normal_rand_vec":
            embs = torch.randn(
                (x.shape[2], x.shape[3]),
                device=x.device,
                dtype=x.dtype,
                generator=self.generator,
            )
            x += embs[None, None]
        elif self.feature_positional_embedding == "uni_rand_vec":
            embs = (
                torch.rand(
                    (x.shape[2], x.shape[3]),
                    device=x.device,
                    dtype=x.dtype,
                    generator=self.generator,
                )
                * 2
                - 1
            )
            x += embs[None, None]
        elif self.feature_positional_embedding == "learned":
            w = self.feature_positional_embedding_embeddings.weight
            embs = w[
                torch.randint(
                    0,
                    w.shape[0],
                    (x.shape[2],),
                    generator=self.generator,
                )
            ]
            x += embs[None, None]
        elif self.feature_positional_embedding == "subspace":
            embs = torch.randn(
                (x.shape[2], x.shape[3] // 4),
                device=x.device,
                dtype=x.dtype,
                generator=self.generator,
            )
            embs = self.feature_positional_embedding_embeddings(embs)
            x += embs[None, None]
        elif self.feature_positional_embedding is None:
            embs = None
        else:
            raise ValueError(f"Unknown {self.feature_positional_embedding=}")

        assert not hasattr(self, "dag_pos_enc_dim") or not self.dag_pos_enc_dim

        return x, y

    def empty_trainset_representation_cache(self) -> None:
        for layer in (self.transformer_decoder or self.transformer_encoder).layers:
            layer.empty_trainset_representation_cache()


def _networkx_add_direct_connections(graph: nx.DiGraph) -> bool:
    added_connection = False
    # Get the list of nodes
    nodes = list(graph.nodes)

    # Iterate over each node
    for node in nodes:
        # Get the direct neighbors of the current node
        neighbors = list(graph.neighbors(node))

        # Iterate over the neighbors of the current node
        for neighbor in neighbors:
            # Get the neighbors of the neighbor
            second_neighbors = list(graph.neighbors(neighbor))

            # Iterate over the neighbors of the neighbor
            for second_neighbor in second_neighbors:
                # Add a direct edge from the current node to the second neighbor,
                # if it doesn't exist already
                if second_neighbor not in graph.neighbors(node):
                    graph.add_edge(node, second_neighbor)

                    added_connection = True
    return added_connection


def _add_pos_emb(
    graph: nx.DiGraph,
    *,
    is_undirected: bool = False,
    k: int = 20,
) -> None:
    from scipy.sparse.linalg import eigs, eigsh

    eig_fn = eigs if not is_undirected else eigsh

    L = nx.directed_laplacian_matrix(graph)
    np.nan_to_num(L, nan=0.0, copy=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eig_vals, eig_vecs = eig_fn(  # type: ignore
            L,
            k=k + 1,
            which="SR" if not is_undirected else "SA",
            return_eigenvectors=True,
        )

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe_ = torch.from_numpy(eig_vecs[:, 1 : k + 1])
        pe = torch.zeros(len(eig_vecs), k)
        pe[:, : pe_.shape[1]] = pe_
        sign = -1 + 2 * torch.randint(0, 2, (k,))
        pe *= sign

        # TODO(old) Double check the ordering is right
        for n, pe_ in zip(graph.nodes(), pe):
            graph.nodes[n]["positional_encoding"] = pe_


class SerializableGenerator(torch.Generator):
    """A serializable version of the torch.Generator, that cna be saved and pickled."""

    def __getstate__(self) -> Any:
        return self.__dict__

    def __setstate__(self, d: Any) -> None:
        self.__dict__ = d
