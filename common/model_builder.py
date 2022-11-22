from encoders.relation_embedding import RelationEmbedding
from encoders.affine_transform import AffineTransform
from decoders.bilinear_diag import BilinearDiag
from extras.graph_representations import Representation

from decoders.nonlinear_transform import NonlinearTransform
from decoders.complex import Complex

from encoders.bipartite_gcn import BipartiteGcn
from encoders.message_gcns.gcn_diag import DiagGcn
from encoders.message_gcns.gcn_basis import BasisGcn
from encoders.message_gcns.gcn_basis_concat import ConcatGcn
from encoders.message_gcns.gcn_basis_stored import BasisGcnStore
from encoders.message_gcns.gcn_basis_plus_diag import BasisGcnWithDiag
from encoders.message_gcns.gcn_basis_times_diag import BasisGcnTimesDiag
# from encoders.message_gcns.gat import gat
from encoders.disen_gcn import Disen_GCN

from encoders.random_vertex_embedding import RandomEmbedding

from extras.residual_layer import ResidualLayer
from extras.highway_layer import HighwayLayer
from extras.dropover import DropoverLayer

from extras.variational_encoding import VariationalEncoding


def build_encoder(encoder_settings, triples):

# 没有经过正则化的权重
    if encoder_settings['Name'] == "embedding":
        input_shape = [int(encoder_settings['EntityCount']),int(encoder_settings['CodeDimension'])]
        embedding = AffineTransform(input_shape,
                                    encoder_settings,
                                    onehot_input=True,
                                    use_bias=False,
                                    use_nonlinearity=False)# embedding =
        full_encoder = RelationEmbedding(input_shape,
                                         encoder_settings,
                                         next_component=embedding)

        return full_encoder  # 返回的full_encoder中codes[0], self.W_relation, codes[2]
    elif encoder_settings['Name'] == "disen_gcn_embedding":
        input_shape = [1713, 1713]
        # print(encoder_settings['channels'])
        # input_parameter = [encoder_settings['in_dim'],
        #                    encoder_settings['channels'],
        #                    encoder_settings['c_dim'], int(encoder_settings['iterations']),
        #                    int(encoder_settings['beta']), int(encoder_settings['layer_num']),
        #                    int(encoder_settings['dropout']), int(encoder_settings['out_dim']), encoder_settings['adj_matrix']]

        embedding = AffineTransform(input_shape,
                                    encoder_settings,
                                    onehot_input=True,
                                    use_bias=False,
                                    use_nonlinearity=False)# embedding =
        dis_embedding = Disen_GCN(encoder_settings, next_component = embedding)

        full_encoder = RelationEmbedding([1713, 64],
                                         encoder_settings,
                                         next_component=dis_embedding)

        return full_encoder# 返回的full_encoder中codes[0], self.W_relation, codes[2]
    elif encoder_settings['Name'] == "gat_embedding":
        # Define graph representation:
        graph = Representation(triples, encoder_settings)

        # Define shapes:
        input_shape = [int(encoder_settings['EntityCount']),
                       int(encoder_settings['InternalEncoderDimension'])]
        internal_shape = [int(encoder_settings['InternalEncoderDimension']),
                          int(encoder_settings['InternalEncoderDimension'])]
        projection_shape = [int(encoder_settings['InternalEncoderDimension']),
                            int(encoder_settings['CodeDimension'])]

        relation_shape = [int(encoder_settings['EntityCount']),
                          int(encoder_settings['CodeDimension'])]

        layers = int(encoder_settings['NumberOfLayers'])

        # Initial embedding:
        encoding = AffineTransform(input_shape,
                                   encoder_settings,
                                   next_component=graph,
                                   onehot_input=True,
                                   use_bias=True,
                                   use_nonlinearity=True)# 初始化嵌入

        # Hidden layers:
        for layer in range(layers):
            use_nonlinearity = layer < layers - 1
            encoding = gat(internal_shape,
                               encoder_settings,
                               next_component=encoding,
                               onehot_input=False,
                               use_nonlinearity=use_nonlinearity)  # 这部分代码为GAT的构建对两层gat进行操作

        # Output transform if chosen:
        if encoder_settings['UseOutputTransform'] == "Yes":
            encoding = AffineTransform(projection_shape,
                                       encoder_settings,
                                       next_component=encoding,
                                       onehot_input=False,
                                       use_nonlinearity=False,
                                       use_bias=True)

        # Encode relations:
        full_encoder = RelationEmbedding(relation_shape,
                                         encoder_settings,
                                         next_component=encoding)# 将关系嵌入加入进来

        return full_encoder


    elif encoder_settings['Name'] == "variational_embedding":
        input_shape = [int(encoder_settings['EntityCount']),
                       int(encoder_settings['CodeDimension'])]

        mu_embedding = AffineTransform(input_shape,
                                    encoder_settings,
                                    onehot_input=True,
                                    use_bias=False,
                                    use_nonlinearity=False)


        sigma_embedding = AffineTransform(input_shape,
                                    encoder_settings,
                                    onehot_input=True,
                                    use_bias=False,
                                    use_nonlinearity=False)

        z = VariationalEncoding(input_shape,
                                encoder_settings,
                                mu_network=mu_embedding,
                                sigma_network=sigma_embedding)

        full_encoder = RelationEmbedding(input_shape,
                                         encoder_settings,
                                         next_component=z)

        return full_encoder

    elif encoder_settings['Name'] == "gcn_diag":
        # Define graph representation:
        graph = Representation(triples, encoder_settings)

        # Define shapes:
        input_shape = [int(encoder_settings['EntityCount']),
                       int(encoder_settings['InternalEncoderDimension'])]
        internal_shape = [int(encoder_settings['InternalEncoderDimension']),
                            int(encoder_settings['InternalEncoderDimension'])]
        projection_shape = [int(encoder_settings['InternalEncoderDimension']),
                            int(encoder_settings['CodeDimension'])]

        relation_shape = [int(encoder_settings['EntityCount']),
                          int(encoder_settings['CodeDimension'])]

        layers = int(encoder_settings['NumberOfLayers'])

        # Initial embedding:
        encoding = AffineTransform(input_shape,
                                    encoder_settings,
                                    next_component=graph,
                                    onehot_input=True,
                                    use_bias=True,
                                    use_nonlinearity=True)

        # Hidden layers:
        for layer in range(layers):
            use_nonlinearity = layer < layers - 1
            encoding = DiagGcn(internal_shape,
                               encoder_settings,
                               next_component=encoding,
                               onehot_input=False,
                               use_nonlinearity=use_nonlinearity)# 这部分代码为GCN的构建

        # Output transform if chosen:
        if encoder_settings['UseOutputTransform'] == "Yes":
            encoding = AffineTransform(projection_shape,
                                       encoder_settings,
                                       next_component=encoding,
                                       onehot_input=False,
                                       use_nonlinearity=False,
                                       use_bias=True)

        # Encode relations:
        full_encoder = RelationEmbedding(relation_shape,
                                         encoder_settings,
                                         next_component=encoding)

        return full_encoder

    elif encoder_settings['Name'] == "gcn_basis":

        # Define graph representation:
        graph = Representation(triples, encoder_settings)

        # Define shapes:
        input_shape = [int(encoder_settings['EntityCount']),
                       int(encoder_settings['InternalEncoderDimension'])]
        internal_shape = [int(encoder_settings['InternalEncoderDimension']),
                          int(encoder_settings['InternalEncoderDimension'])]
        projection_shape = [int(encoder_settings['InternalEncoderDimension']),
                            int(encoder_settings['CodeDimension'])]

        relation_shape = [int(encoder_settings['EntityCount']),
                          int(encoder_settings['CodeDimension'])]

        layers = int(encoder_settings['NumberOfLayers'])

        # Initial embedding:
        if encoder_settings['UseInputTransform'] == "Yes":
            encoding = AffineTransform(input_shape,
                                       encoder_settings,
                                       next_component=graph,
                                       onehot_input=True,
                                       use_bias=True,
                                       use_nonlinearity=True)
        elif encoder_settings['RandomInput'] == 'Yes':
            encoding = RandomEmbedding(input_shape,
                                       encoder_settings,
                                       next_component=graph)
        elif encoder_settings['PartiallyRandomInput'] == 'Yes':
            encoding1 = AffineTransform(input_shape,
                                       encoder_settings,
                                       next_component=graph,
                                       onehot_input=True,
                                       use_bias=True,
                                       use_nonlinearity=False)
            encoding2 = RandomEmbedding(input_shape,
                                       encoder_settings,
                                       next_component=graph)
            encoding = DropoverLayer(input_shape,
                                     next_component=encoding1,
                                     next_component_2=encoding2)
        else:
            encoding = graph

        # Hidden layers:
        encoding = apply_basis_gcn(encoder_settings, encoding, internal_shape, layers)

        # Output transform if chosen:
        if encoder_settings['UseOutputTransform'] == "Yes":
            encoding = AffineTransform(projection_shape,
                                       encoder_settings,
                                       next_component=encoding,
                                       onehot_input=False,
                                       use_nonlinearity=False,
                                       use_bias=True)

        # Encode relations:
        full_encoder = RelationEmbedding(relation_shape,
                                         encoder_settings,
                                         next_component=encoding)

        return full_encoder

    elif encoder_settings['Name'] == "variational_gcn_basis":
        graph = Representation(triples, encoder_settings)

        # Define graph representation:
        graph = Representation(triples, encoder_settings)

        # Define shapes:
        input_shape = [int(encoder_settings['EntityCount']),
                       int(encoder_settings['InternalEncoderDimension'])]
        internal_shape = [int(encoder_settings['InternalEncoderDimension']),
                          int(encoder_settings['InternalEncoderDimension'])]
        projection_shape = [int(encoder_settings['InternalEncoderDimension']),
                            int(encoder_settings['CodeDimension'])]

        relation_shape = [int(encoder_settings['EntityCount']),
                          int(encoder_settings['CodeDimension'])]

        layers = int(encoder_settings['NumberOfLayers'])

        # Initial embedding:
        if encoder_settings['UseInputTransform'] == "Yes":
            encoding = AffineTransform(input_shape,
                                       encoder_settings,
                                       next_component=graph,
                                       onehot_input=True,
                                       use_bias=True,
                                       use_nonlinearity=True)
        else:
            encoding = graph

        # Hidden layers:
        encoding = apply_basis_gcn(encoder_settings, encoding, internal_shape, layers)

        mu_encoding = AffineTransform(projection_shape,
                                       encoder_settings,
                                       next_component=encoding,
                                       onehot_input=False,
                                       use_nonlinearity=False,
                                       use_bias=True)

        sigma_encoding = AffineTransform(projection_shape,
                                       encoder_settings,
                                       next_component=encoding,
                                       onehot_input=False,
                                       use_nonlinearity=False,
                                       use_bias=True)
        #mu_encoding = apply_basis_gcn(encoder_settings, encoding, internal_shape, layers)
        #sigma_encoding = apply_basis_gcn(encoder_settings, encoding, internal_shape, layers)

        encoding = VariationalEncoding(input_shape,
                                encoder_settings,
                                mu_network=mu_encoding,
                                sigma_network=sigma_encoding)

        # Output transform if chosen:
        if encoder_settings['UseOutputTransform'] == "Yes":
            encoding = AffineTransform(projection_shape,
                                       encoder_settings,
                                       next_component=encoding,
                                       onehot_input=False,
                                       use_nonlinearity=False,
                                       use_bias=True)

        # Encode relations:
        full_encoder = RelationEmbedding(relation_shape,
                                         encoder_settings,
                                         next_component=encoding)

        return full_encoder

    else:
        '''
        elif encoder_settings['Name'] == "bipartite_gcn":
            graph = Representation(triples, encoder_settings, bipartite=True)

            first_layer = BipartiteGcn(encoder_settings, graph)
            second_layer = BipartiteGcn(encoder_settings, graph, next_component=first_layer)
            third_layer = BipartiteGcn(encoder_settings, graph, next_component=second_layer)
            fourth_layer = BipartiteGcn(encoder_settings, graph, next_component=third_layer)

            transform = AffineTransform(fourth_layer, encoder_settings)

            return RelationEmbedding(transform, encoder_settings)
        '''
        return None


def apply_basis_gcn(encoder_settings, encoding, internal_shape, layers):
    for layer in range(layers):
        use_nonlinearity = layer < layers - 1

        if layer == 0 \
                and encoder_settings['UseInputTransform'] == "No" \
                and encoder_settings['RandomInput'] == "No"  \
                and encoder_settings['PartiallyRandomInput'] == "No" :
            onehot_input = True
        else:
            onehot_input = False

        if encoder_settings['AddDiagonal'] == "Yes":
            model = BasisGcnWithDiag
        elif encoder_settings['DiagonalCoefficients'] == "Yes":
            model = BasisGcnTimesDiag
        elif encoder_settings['StoreEdgeData'] == "Yes":
            model = BasisGcnStore
        elif 'Concatenation' in encoder_settings and encoder_settings['Concatenation'] == "Yes":
            model = ConcatGcn
        else:
            model = BasisGcn

        new_encoding = model(internal_shape,
                             encoder_settings,
                             next_component=encoding,
                             onehot_input=onehot_input,
                             use_nonlinearity=use_nonlinearity)

        if encoder_settings['SkipConnections'] == 'Residual' and onehot_input == False:
            encoding = ResidualLayer(internal_shape, next_component=new_encoding, next_component_2=encoding)
        if encoder_settings['SkipConnections'] == 'Highway' and onehot_input == False:
            encoding = HighwayLayer(internal_shape, next_component=new_encoding, next_component_2=encoding)
        else:
            encoding = new_encoding

    return encoding


def build_decoder(encoder, decoder_settings):
    if decoder_settings['Name'] == "bilinear-diag":
        return BilinearDiag(encoder, decoder_settings)
    elif decoder_settings['Name'] == "complex":
        return Complex(int(decoder_settings['CodeDimension']), decoder_settings, next_component=encoder)
    elif decoder_settings['Name'] == "nonlinear":
        return NonlinearTransform(encoder, decoder_settings)
    else:
        return None