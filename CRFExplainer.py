import numpy as np
import time
import copy
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
import multiprocessing
from multiprocessing import Pool


import shap
from lime.lime_tabular import LimeTabularExplainer



class CRFCFExplainer:
    def __init__(self, crf, train_set, feature_names, feature_cons=None, dist_type='L2'):
        self.model = crf
        self.rf = crf.model
        self.s = crf.s
        self.n_trees = crf.n_trees
        self.classes = crf.model.classes_
        self.n_classes = len(self.classes)
        self.train_set = train_set
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.immutable_features = feature_cons['immutable']
        self.increasing_features = feature_cons['increasing']
        self.decreasing_features = feature_cons['decreasing']
        self.feature_types = feature_cons['data types']
        self.dist_type = dist_type
        self.epsilon = 1e-4
        self.lof = LocalOutlierFactor(novelty=True)

        # feature-importance related state
        self.global_fi_values_ = None      # global feature importance vector
        self.global_fi_method = None       # name of the method used

    @staticmethod
    def _process_tree(tree, n_features, n_classes, train_max, s):
        """
        Parallel processing of a single tree:
        - build decision paths (intervals for each feature)
        - build probability intervals (same as original CRF code)
        - build per-tree decision_paths_dict
        """
        decision_paths_dict_part = {}
        decision_paths_part = []
        probability_intervals_part = []

        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        value = tree.tree_.value.reshape((-1, n_classes))

        # initial region: [-0.0001, max(train)] for each feature
        dp_stack = [[[-0.0001, 1.0] for _ in range(n_features)]]
        for d in range(n_features):
            dp_stack[0][d][1] = train_max[d]

        for i in range(n_nodes):
            is_internal_node = (children_left[i] != children_right[i])
            if is_internal_node:
                parent_dp = dp_stack.pop()
                left_dp = copy.deepcopy(parent_dp)
                right_dp = copy.deepcopy(parent_dp)

                # right child first on stack
                dp_stack.append(right_dp)
                dp_stack.append(left_dp)

                # left: upper bound changed
                dp_stack[-1][feature[i]][1] = threshold[i]
                # right: lower bound changed
                dp_stack[-2][feature[i]][0] = threshold[i]
            else:
                # probabilities: exactly the same logic as original CRFCFExplainer.fit
                N_S = float(np.sum(value[i]) + s)
                n_samples = value[i].repeat(2)
                n_samples[1:2 * n_classes:2] = n_samples[1:2 * n_classes:2] + s
                probabilities = (n_samples / N_S).round(4)

                dp_to_add = np.array(dp_stack.pop()).flatten()
                decision_paths_dict_part[i] = dp_to_add
                decision_paths_part.append(dp_to_add)
                probability_intervals_part.append(probabilities)

        return decision_paths_dict_part, decision_paths_part, probability_intervals_part

    @staticmethod
    def _process_leaf(args):
        """
        Parallel processing of one row of leaves_index to build a live region.
        """
        i, leaves_index, decision_paths_dict, n_trees, n_features = args
        current_live_region = np.zeros((n_trees, 2 * n_features))
        for t in range(n_trees):
            current_live_region[t] = decision_paths_dict[t][leaves_index[i, t]]

        live_region = np.zeros(2 * n_features)
        live_region[0::2] = current_live_region[:, 0::2].max(axis=0)
        live_region[1::2] = current_live_region[:, 1::2].min(axis=0)
        return live_region

    # ------------------------------------------------------------------
    # Utilities for feature importance
    # ------------------------------------------------------------------
    def _check_imp2_available(self):
        """Ensure that the CRF model provides an 'imp2' method."""
        if not hasattr(self.model, "imp2"):
            raise AttributeError(
                "The CRF model does not implement an 'imp2' method.\n"
                "Please add 'imp2(self, X)' that returns Imp2(h, x)."
            )

    # ---------------- Global FI ----------------
    def _compute_global_fi_mdi(self):
        """MDI-based global feature importance (standard RF feature_importances_)."""
        fi = np.asarray(self.rf.feature_importances_, dtype=float)
        return fi


    def _compute_global_fi_pfi(self,
                           random_state=None,
                           max_samples_x=200,
                           max_samples_z=200):
        """
        Global permutation feature importance based on Imp2, following Eq. (27).
    
        Φ(j | h) ≈ E_X[ φ(j | h, X) ],
        where φ(j | h, x) = Imp2(h, x) - E[Imp2(h, X_~j) | x].
        A larger positive value means that, on average, modifying feature j
        can reduce indeterminacy.
        """
        self._check_imp2_available()
    
        X = self.train_set
        if not hasattr(self, "train_predictions"):
            self.train_predictions = self.model.predict(self.train_set)
        y_hat = self.train_predictions
    
        # Use only indeterminate predictions, as in the paper
        mask_indet = (y_hat == -1)
        X_used = X[mask_indet] if mask_indet.any() else X
    
        # Subsample at most max_samples_x instances for efficiency
        rng = np.random.RandomState(random_state)
        if X_used.shape[0] > max_samples_x:
            idx = rng.choice(X_used.shape[0], size=max_samples_x, replace=False)
            X_used = X_used[idx]
    
        n_features = self.n_features
        fi = np.zeros(n_features, dtype=float)
    
        # Average local PFI over selected instances
        for i, x in tqdm(enumerate(X_used)):
            # You can vary the random_state here if you like
            phi_local = self._local_pfi(
                x,
                random_state=random_state,
                max_samples=max_samples_z
            )
            fi += phi_local
    
        fi /= float(X_used.shape[0])
    
        return fi


    def _compute_global_fi_shap(self,
                                random_state=None,
                                max_background=50,
                                shap_nsamples=200):
        """
        SHAP-FI global importance based on Imp2.

        Φ(j | h) = mean_{x with indeterminate prediction} |φ(j | h, x)|
        where φ are Kernel SHAP values of f(x)=Imp2(h,x).
        """
        if shap is None:
            raise ImportError(
                "shap is not installed. Please 'pip install shap' to use SHAP-FI-Global."
            )

        self._check_imp2_available()

        X = self.train_set
        if not hasattr(self, "train_predictions"):
            self.train_predictions = self.model.predict(self.train_set)
        y_hat = self.train_predictions

        mask_indet = (y_hat == -1)
        X_used = X[mask_indet] if mask_indet.any() else X

        # Use simple random sampling to select a small background set.
        rng = np.random.RandomState(random_state)
        if X_used.shape[0] > max_background:
            idx = rng.choice(X_used.shape[0], size=max_background, replace=False)
            X_bg = X_used[idx]
        else:
            X_bg = X_used

        f = lambda data: self.model.imp2(np.asarray(data))

        explainer = shap.KernelExplainer(f, X_bg)
        shap_values = explainer.shap_values(X_bg, nsamples=shap_nsamples)
        fi = np.mean(np.abs(shap_values), axis=0)
        return fi



    def compute_global_feature_importance(self,
                                          method='MDI-Global',
                                          **kwargs):
        """
        Public API to compute global feature importance.

        method ∈ {'MDI-Global', 'PFI-Global', 'SHAP-FI-Global'}
        """
        if method == 'MDI-Global':
            fi = self._compute_global_fi_mdi()
        elif method == 'PFI-Global':
            fi = self._compute_global_fi_pfi(**kwargs)
        elif method == 'SHAP-FI-Global':
            fi = self._compute_global_fi_shap(**kwargs)
        else:
            raise ValueError(f"Unknown global feature importance method: {method}")

        self.global_fi_method = method
        self.global_fi_values_ = fi
        
        # print(fi)
        return fi

    # ---------------- Local FI ----------------
    def _local_pfi(self, x, random_state=None, max_samples=200):
        """
        Local permutation feature importance (PFI-Local) based on Imp2.

        φ(j | h, x) ≈ Imp2(h, x) - E[Imp2(h, X_~j)]
        where X_~j are samples obtained by replacing x_j with values sampled
        from the empirical marginal of feature j in the training set.
        """
        self._check_imp2_available()

        X = self.train_set
        rng = np.random.RandomState(random_state)

        x = np.asarray(x).reshape(1, -1)
        baseline = float(self.model.imp2(x)[0])
        x0 = x[0]
        n_features = self.n_features
        phi = np.zeros(n_features, dtype=float)

        for j in range(n_features):
            col = X[:, j]
            if col.shape[0] > max_samples:
                idx = rng.choice(col.shape[0], size=max_samples, replace=False)
                values = col[idx]
            else:
                values = col

            X_pert = np.repeat(x0.reshape(1, -1), values.shape[0], axis=0)
            X_pert[:, j] = values
            imp_pert = self.model.imp2(X_pert)
            phi[j] = baseline - float(imp_pert.mean())

        return phi


    def _local_shap(self,
                    x,
                    random_state=None,
                    max_background=50,
                    shap_nsamples=200):
        """
        Local SHAP on f(x)=Imp2(h,x).
        """
        if shap is None:
            raise ImportError(
                "shap is not installed. Please 'pip install shap' to use SHAP-Local."
            )

        self._check_imp2_available()

        X_bg = self.train_set
        rng = np.random.RandomState(random_state)
        if X_bg.shape[0] > max_background:
            idx = rng.choice(X_bg.shape[0], size=max_background, replace=False)
            X_bg = X_bg[idx]

        f = lambda data: self.model.imp2(np.asarray(data))

        explainer = shap.KernelExplainer(f, X_bg)
        shap_values = explainer.shap_values(x.reshape(1, -1), nsamples=shap_nsamples, silent=True)
        return shap_values[0]

    
    

    def _local_lime(self, x, num_samples=200):
        """
        Local LIME on f(x) = Imp2(h, x).
    
        Returns a non-negative importance vector where
        larger values mean a stronger local influence of the
        feature on Imp2 around x.
        """
        if LimeTabularExplainer is None:
            raise ImportError(
                "lime is not installed. Please 'pip install lime' to use LIME-Local."
            )
    
        self._check_imp2_available()
    
        explainer = LimeTabularExplainer(
            self.train_set,
            mode='regression',
            feature_names=self.feature_names,
            discretize_continuous=False
        )
    
        f = lambda data: self.model.imp2(np.asarray(data))
        exp = explainer.explain_instance(
            x,
            f,
            num_features=self.n_features,
            num_samples=num_samples
        )
    
        phi_raw = np.zeros(self.n_features, dtype=float)
        # exp.local_exp[0] contains (feature_id, weight) tuples
        for fid, w in exp.local_exp[0]:
            phi_raw[fid] = w
    
        # Option 1: importance = absolute influence on Imp2
        fi = np.abs(phi_raw)
    
        # If you prefer "only influence that tends to reduce Imp2":
        # fi = np.maximum(-phi_raw, 0.0)
    
        return fi

    def compute_local_feature_importance(self, x, method='PFI-Local', **kwargs):
        """
        Public API to compute local feature importance for a single instance x.

        method ∈ {'PFI-Local', 'SHAP-Local', 'LIME-Local'}
        """
        x = np.asarray(x).ravel()

        if method == 'PFI-Local':
            return self._local_pfi(x, **kwargs)
        elif method == 'SHAP-Local':
            return self._local_shap(x, **kwargs)
        elif method == 'LIME-Local':
            return self._local_lime(x, **kwargs)
        else:
            raise ValueError(f"Unknown local feature importance method: {method}")

    # ------------------------------------------------------------------
    # Fit: now includes global FI computation
    # ------------------------------------------------------------------
    def fit(self, global_fi_method='MDI-Global', **global_fi_kwargs):
        # predictions on training data (for MO and FI)
        self.train_predictions = self.model.predict(self.train_set)

        # 1) Global feature importance and feature ordering
        fi = self.compute_global_feature_importance(
            method=global_fi_method,
            **global_fi_kwargs
        )
        # According to the paper, more important features should be deeper,
        # so we sort indices in ascending order of importance.
        self.feature_order = np.argsort(fi)
        
        # feture order for DisCERN method
        self.discern_feature_order = np.argsort(-self.rf.feature_importances_)
        
        # distance adjustments
        self.dist_std = np.std(self.train_set, axis=0)
        medians_abs_diff = abs(self.train_set - np.median(self.train_set, axis=0))
        self.dist_mad = np.mean(medians_abs_diff, axis=0)
        self.dist_mad[self.dist_mad == 0] = 0.5

        # train LOF
        self.lof.fit(self.train_set)

        # ===== 1) parallel extraction of decision paths & probability intervals =====
        train_max = self.train_set.max(axis=0)

        print("Building decision paths and probability intervals with parallel processing...")
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(
                self._process_tree,
                [
                    (tree, self.n_features, self.n_classes, train_max, self.s)
                    for tree in self.rf.estimators_
                ]
            )

        # aggregate per-tree results
        self.decision_paths_dict = {}
        decision_paths_list = []
        probability_intervals_list = []

        for t, (dict_part, dp_part, prob_part) in enumerate(results):
            self.decision_paths_dict[t] = dict_part
            if dp_part:
                decision_paths_list.extend(dp_part)
                probability_intervals_list.extend(prob_part)

        if len(decision_paths_list) > 0:
            self.decision_paths = np.vstack(decision_paths_list)
            self.probability_intervals = np.vstack(probability_intervals_list)
        else:
            # degenerate fallback, should not really happen
            self.decision_paths = np.empty((0, 2 * self.n_features))
            self.probability_intervals = np.empty((0, 2 * self.n_classes))

        # ===== 2) extract split points, split intervals, intervals_with_leaves =====
        self.split_points = {}
        self.split_intervals = {}
        self.intervals_with_leaves = {}

        for d in range(self.n_features):
            # all bounds for feature d from decision paths
            split_points_d = np.unique(self.decision_paths[:, 2 * d:2 * d + 2].reshape((1, -1)))
            split_points_d.sort()
            self.split_points[d] = split_points_d

            if len(split_points_d) > 1:
                split_intervals_d = np.zeros((len(split_points_d) - 1, 2))
                split_intervals_d[:, 0] = split_points_d[:-1]
                split_intervals_d[:, 1] = split_points_d[1:]
            else:
                split_intervals_d = np.zeros((0, 2))

            self.split_intervals[d] = split_intervals_d

            # build mapping: split interval -> leaves indices
            self.intervals_with_leaves[d] = {}
            decision_path_d = self.decision_paths[:, 2 * d:2 * d + 2]
            for i in range(len(split_intervals_d)):
                split_interval = (split_intervals_d[i, 0], split_intervals_d[i, 1])
                idx = np.where(
                    np.maximum(split_interval[0], decision_path_d[:, 0])
                    < np.minimum(split_interval[1], decision_path_d[:, 1])
                )[0]
                self.intervals_with_leaves[d][split_interval] = idx

        # ===== 3) parallel construction of live regions =====
        leaves_index = self.rf.apply(self.train_set)
        leaves_index = np.unique(leaves_index, axis=0)
        self.leaves_index = leaves_index  # optional, but sometimes useful to keep

        print("Building live regions with parallel processing...")
        args = [
            (i, leaves_index, self.decision_paths_dict, self.n_trees, self.n_features)
            for i in range(len(leaves_index))
        ]

        # Limit number of processes to avoid overhead on small datasets
        with Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
            live_regions_list = pool.map(self._process_leaf, args)

        self.live_regions = np.array(live_regions_list)

        # cautious RF prediction for each live region (center of region)
        centers = (self.live_regions[:, 0::2] + self.live_regions[:, 1::2]) / 2.0
        self.live_regions_predictions = self.model.predict(centers)


    def __instance_dist(self, x, X, dist_type):
        if dist_type == 'L1':
            dists = (abs(X - x) / self.dist_mad).sum(axis=1)
        elif dist_type == 'L0':
            dists = (X != x).sum(axis=1)
        else:
            dists = np.sqrt(((X - x) ** 2 / self.dist_std).sum(axis=1))
        return dists

    def __interval_dist(self, d, x_d, intervals, dist_type):
        in_index = (intervals[:, 0] < x_d) * (x_d <= intervals[:, 1])
        left_index = intervals[:, 1] < x_d
        right_index = intervals[:, 0] >= x_d
        dists = np.zeros(len(intervals))
        dists[in_index] = 0
        if self.feature_types is not None and self.feature_types[d] == 'int':
            dists[left_index] = x_d - np.floor(intervals[left_index, 1])
            dists[right_index] = np.ceil(intervals[right_index, 1] + self.epsilon) - x_d
        else:
            dists[left_index] = x_d - intervals[left_index, 1]
            dists[right_index] = intervals[right_index, 1] - x_d + self.epsilon
        
        if dist_type == 'L1':
            dists = dists / self.dist_mad[d]
        elif dist_type == 'L0':
            dists = (dists != 0) * 1
        else:
            dists = dists ** 2 / self.dist_std[d]

        return dists

    def __generate_cf_in_regions(self, x, regions):
        candidates = x.reshape((1, -1)).repeat(len(regions), axis=0)
        for d in range(self.n_features):
            take_inf = regions[:, 2 * d] >= x[d]
            take_sup = regions[:, 2 * d + 1] < x[d]
            if self.feature_types is not None and self.feature_types[d] == 'int':
                inf_values = np.ceil(regions[take_inf, 2 * d] + self.epsilon)
                sup_values = np.floor(regions[take_sup, 2 * d + 1])
            else:
                inf_values = regions[take_inf, 2 * d] + self.epsilon
                sup_values = regions[take_sup, 2 * d + 1]

            candidates[take_inf, d] = inf_values
            candidates[take_sup, d] = sup_values

        return candidates

    def mo(self, x, target, dist_type=None):
        """Minimum Observable counterfactual for cautious random forests."""
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None

        # choose distance type
        if dist_type is None:
            dist_type = self.dist_type if self.dist_type is not None else 'L1'

        remain_instances = self.train_set[self.train_predictions == target].copy()

        # apply feature constraints
        for d in range(self.n_features):
            feature = self.feature_names[d]

            if self.immutable_features is not None and feature in self.immutable_features:
                index = (x[d] == remain_instances[:, d])
                remain_instances = remain_instances[index]

            if self.increasing_features is not None and feature in self.increasing_features:
                index = (x[d] <= remain_instances[:, d])
                remain_instances = remain_instances[index]

            if self.decreasing_features is not None and feature in self.decreasing_features:
                index = (x[d] >= remain_instances[:, d])
                remain_instances = remain_instances[index]

            if len(remain_instances) == 0:
                break

        if len(remain_instances) == 0:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None

        dists = self.__instance_dist(x, remain_instances, dist_type)
        cf_index = np.argmin(dists)
        min_dist = float(dists[cf_index])
        cf = remain_instances[cf_index]

        return cf, round(min_dist, 4)

    def discern(self, x, target, dist_type=None):
        """DisCERN-style counterfactual: sparsify an initial MO counterfactual."""
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None

        if dist_type is None:
            dist_type = self.dist_type if self.dist_type is not None else 'L1'

        init_cf, init_min_dist = self.mo(x, target, dist_type)
        if init_cf is None:
            return None, None

        cf = x.copy()
        for d in self.discern_feature_order:
            cf[d] = init_cf[d]
            prediction = self.model.predict(cf.reshape((1, -1)))[0]
            if prediction == target:
                min_dist = float(self.__instance_dist(x, cf.reshape((1, -1)), dist_type)[0])
                return cf, round(min_dist, 4)

        # fallback: return the initial MO counterfactual
        return init_cf, round(float(init_min_dist), 4)

    def ofcc(self, x, target, dist_type=None):
        """One-Feature-Changed Counterfactual for cautious random forests."""
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None

        if dist_type is None:
            dist_type = self.dist_type if self.dist_type is not None else 'L1'

        cf = None
        min_dist = 1e8

        for d in range(self.n_features):
            feature = self.feature_names[d]
            split_points_d = self.split_points[d]
            if split_points_d is None or len(split_points_d) == 0:
                continue

            candidates = None

            # immutable feature: cannot be changed
            if self.immutable_features is not None and feature in self.immutable_features:
                continue

            # feature can only increase
            if self.increasing_features is not None and feature in self.increasing_features:
                pts = split_points_d[split_points_d > x[d]]
                if len(pts) > 0:
                    candidates = x.reshape((1, -1)).repeat(len(pts), axis=0)
                    if self.feature_types is not None and self.feature_types[d] == 'int':
                        candidates[:, d] = np.ceil(pts + self.epsilon)
                    else:
                        candidates[:, d] = pts + self.epsilon

            # feature can only decrease
            if self.decreasing_features is not None and feature in self.decreasing_features:
                pts = split_points_d[split_points_d < x[d]]
                if len(pts) > 0:
                    candidates = x.reshape((1, -1)).repeat(len(pts), axis=0)
                    if self.feature_types is not None and self.feature_types[d] == 'int':
                        candidates[:, d] = np.floor(pts)
                    else:
                        candidates[:, d] = pts

            # unconstrained feature: can move in both directions
            if (self.increasing_features is None or feature not in self.increasing_features) and \
               (self.decreasing_features is None or feature not in self.decreasing_features) and \
               (self.immutable_features is None or feature not in self.immutable_features):
                pts = split_points_d
                candidates = x.reshape((1, -1)).repeat(len(pts), axis=0)
                index1 = pts > x[d]
                index2 = pts < x[d]
                if self.feature_types is not None and self.feature_types[d] == 'int':
                    candidates[index1, d] = np.ceil(pts[index1] + self.epsilon)
                    candidates[index2, d] = np.floor(pts[index2])
                else:
                    candidates[index1, d] = pts[index1] + self.epsilon
                    candidates[index2, d] = pts[index2]

            if candidates is None:
                continue

            predictions = self.model.predict(candidates)
            remain_instances = candidates[predictions == target]

            if len(remain_instances) == 0:
                continue

            dists = self.__instance_dist(x, remain_instances, dist_type)
            min_index = np.argmin(dists)

            if dists[min_index] < min_dist:
                min_dist = float(dists[min_index])
                cf = remain_instances[min_index]

        if cf is None:
            # print("Can't get one-feature-changed counterfactual example!")
            return self.discern(x, target, dist_type)
        else:
            return cf, round(min_dist, 4)

    def lire(self, x, target, dist_type=None):
        """Live-Region based counterfactual search for cautious random forests."""
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None

        if dist_type is None:
            dist_type = self.dist_type if self.dist_type is not None else 'L1'

        live_regions = self.live_regions.copy()

        # apply feature constraints to prune live regions
        for d in range(self.n_features):
            feature = self.feature_names[d]

            if self.immutable_features is not None and feature in self.immutable_features:
                index = (live_regions[:, 2 * d] < x[d]) * (x[d] <= live_regions[:, 2 * d + 1])
                live_regions = live_regions[index]

            if self.increasing_features is not None and feature in self.increasing_features:
                index = live_regions[:, 2 * d + 1] > x[d]
                live_regions = live_regions[index]

            if self.decreasing_features is not None and feature in self.decreasing_features:
                index = live_regions[:, 2 * d] < x[d]
                live_regions = live_regions[index]

            if len(live_regions) == 0:
                break

        if len(live_regions) == 0:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None

        candidates = self.__generate_cf_in_regions(x, live_regions)
        candidates = candidates[self.live_regions_predictions == target]

        if len(candidates) == 0:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None

        dists = self.__instance_dist(x, candidates, dist_type)
        cf_index = np.argmin(dists)
        min_dist = float(dists[cf_index])
        cf = candidates[cf_index]

        return cf, round(min_dist, 4)

    def eece(self, x, target, dist_type=None):
        """EECE-style counterfactual combining decision paths and live regions."""
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None

        if dist_type is None:
            dist_type = self.dist_type if self.dist_type is not None else 'L1'

        cf, min_dist = self.lire(x, target, dist_type)

        regions = np.concatenate((self.decision_paths.copy(), self.live_regions.copy()), axis=0)

        # apply feature constraints to prune regions
        for d in range(self.n_features):
            feature = self.feature_names[d]

            if self.immutable_features is not None and feature in self.immutable_features:
                index = (regions[:, 2 * d] < x[d]) * (x[d] <= regions[:, 2 * d + 1])
                regions = regions[index]

            if self.increasing_features is not None and feature in self.increasing_features:
                index = regions[:, 2 * d + 1] > x[d]
                regions = regions[index]

            if self.decreasing_features is not None and feature in self.decreasing_features:
                index = regions[:, 2 * d] < x[d]
                regions = regions[index]

            if len(regions) == 0:
                break

        if len(regions) == 0:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None

        candidates = self.__generate_cf_in_regions(x, regions)
        dists = self.__instance_dist(x, candidates, dist_type)

        idx = (dists <= min_dist)
        candidates = candidates[idx]
        dists = dists[idx]

        if len(candidates) == 0:
            return cf, min_dist
        else:
            predictions = self.model.predict(candidates)
            idx = predictions == target
            candidates = candidates[idx]
            dists = dists[idx]

        if len(candidates) == 0:
            return cf, min_dist
        else:
            plausibility = self.lof.predict(candidates)
            idx = (plausibility == 1)
            candidates = candidates[idx]
            dists = dists[idx]

        if len(candidates) == 0:
            return cf, min_dist
        cf_index = np.argmin(dists)
        min_dist = dists[cf_index]
        cf = candidates[cf_index]

        return cf, round(min_dist, 4)

    def __filter_split_intervals(self, x, init_dist, dist_type):
        """Filter split intervals per feature under actionability and distance budget."""
        split_intervals = copy.deepcopy(self.split_intervals)
        n_intervals = np.zeros(self.n_features)
        split_interval_dists = {}

        for d in range(self.n_features):
            feature = self.feature_names[d]
            split_intervals_d = split_intervals[d]

            # immutable features: keep only the interval containing x[d]
            if self.immutable_features is not None and feature in self.immutable_features:
                index = (split_intervals_d[:, 0] < x[d]) * (x[d] <= split_intervals_d[:, 1])
                split_intervals_d = split_intervals_d[index]
                split_intervals[d] = split_intervals_d
                n_intervals[d] = len(split_intervals_d)
                continue

            # compute distances to all candidate intervals
            dists = self.__interval_dist(d, x[d], split_intervals_d, dist_type)
            interval_order = np.argsort(dists)
            dists = dists[interval_order]
            split_intervals_d = split_intervals_d[interval_order]

            # feature can only increase
            if self.increasing_features is not None and feature in self.increasing_features:
                mask = split_intervals_d[:, 1] >= x[d]
                split_intervals_d = split_intervals_d[mask]
                dists = dists[mask]

            # feature can only decrease
            if self.decreasing_features is not None and feature in self.decreasing_features:
                mask = split_intervals_d[:, 0] < x[d]
                split_intervals_d = split_intervals_d[mask]
                dists = dists[mask]

            # keep only intervals within the current distance upper bound
            index = dists <= init_dist
            split_intervals_d = split_intervals_d[index]
            dists = dists[index]

            n_intervals[d] = len(split_intervals_d)
            split_interval_dists[d] = dists
            split_intervals[d] = split_intervals_d

        return split_intervals, split_interval_dists, n_intervals
    

    def exact_cf(self, x, target, dist_type=None, feature_order=None):
        """Exact branch-and-bound counterfactual search tailored to cautious random forests.

        Uses the OFCC/DiSCERN solution as an initial upper bound and then explores
        combinations of split intervals, with LOF-based plausibility filtering.

        feature_order:
            - if provided, use it directly
            - else: use np.arange(self.n_features)
        """
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None

        # choose distance type
        if dist_type is None:
            dist_type = self.dist_type if self.dist_type is not None else 'L1'

        # decide which feature ordering to use
        if feature_order is None:
            feature_order = np.arange(self.n_features)
        else:
            feature_order = np.asarray(feature_order, dtype=int)

        start_time = time.time()

        # initial solution from OFCC (used as an upper bound)
        init_cf, init_dist = self.ofcc(x, target, dist_type)
        if init_cf is None:
            init_cf, init_dist = self.discern(x, target, dist_type)
        if init_cf is None or init_dist is None:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None

        # for L2, work with squared distance inside the search
        if dist_type == 'L2':
            init_dist = init_dist ** 2

        # filter split intervals according to the current distance upper bound
        split_intervals, split_interval_dists, n_intervals = self.__filter_split_intervals(
            x, init_dist, dist_type
        )

        # ---------- (1) Precompute per-feature minimum distance & suffix lower bounds ----------
        # For each feature, keep the minimum distance among its allowed intervals
        min_dists_per_feature = np.zeros(self.n_features)
        for d in range(self.n_features):
            dists_d = split_interval_dists.get(d, None)
            if dists_d is not None and len(dists_d) > 0:
                min_dists_per_feature[d] = dists_d[0]
            else:
                min_dists_per_feature[d] = 0.0

        # Reorder these minima according to feature_order
        min_dists_ordered = min_dists_per_feature[feature_order]

        # suffix_min[i] = lower bound on the minimal additional distance coming
        # from levels i, i+1, ..., end, assuming we always choose the best interval
        suffix_min = np.zeros(self.n_features + 1)
        for i in range(self.n_features - 1, -1, -1):
            suffix_min[i] = suffix_min[i + 1] + min_dists_ordered[i]
        # -------------------------------------------------------------------------

        n_checked_intervals = np.zeros(self.n_features, dtype='int64')
        current_level = 0
        current_region = np.zeros(2 * self.n_features)
        feature_dists = np.zeros(self.n_features)
        leaf_index = {}

        cf = init_cf.copy()
        min_dist = float(init_dist)

        while True:
            end_time = time.time()
            # stop when backtracked above the root or time budget exceeded
            if current_level < 0 or (end_time - start_time) > 6:
                if dist_type == 'L2':
                    min_dist = np.sqrt(min_dist)
                return cf, round(float(min_dist), 4)

            d = int(feature_order[current_level])

            # if no intervals for this feature or all intervals checked → backtrack
            if n_intervals[d] == 0 or n_checked_intervals[d] == n_intervals[d]:
                n_checked_intervals[d] = 0
                feature_dists[d] = 0.0
                current_level -= 1
                continue

            # take the next interval distance for feature d
            feature_dists[d] = split_interval_dists[d][n_checked_intervals[d]]

            # ---------- (2) Stronger branch-and-bound pruning with lower bound ----------
            # partial distance from already fixed features
            current_partial = feature_dists.sum()
            # theoretical minimal additional distance from remaining features
            # (levels deeper than current_level)
            remaining_min = suffix_min[current_level + 1]

            # if even in the best case this branch cannot beat current min_dist → prune
            if current_partial + remaining_min >= min_dist:
                n_checked_intervals[d] = 0
                feature_dists[d] = 0.0
                current_level -= 1
                continue
            # --------------------------------------------------------------------------

            # choose the actual interval for this feature
            interval = split_intervals[d][int(n_checked_intervals[d])]
            interval_key = (interval[0], interval[1])
            n_checked_intervals[d] += 1
            current_region[2 * d:2 * d + 2] = interval

            # update the set of leaves compatible with the current partial region
            if current_level == 0:
                leaf_index[current_level] = self.intervals_with_leaves[d][interval_key]
            else:
                leaf_index[current_level] = np.intersect1d(
                    leaf_index[current_level - 1],
                    self.intervals_with_leaves[d][interval_key]
                )

            # if all features have been fixed → evaluate the candidate
            if current_level == self.n_features - 1:
                candidate = self.__generate_cf_in_regions(
                    x, current_region.reshape((1, -1))
                )
                current_prediction = self.model.predict(candidate.reshape((1, -1)))[0]

                # update best solution if prediction matches target and LOF deems it plausible
                if current_prediction == target:
                    if self.lof.predict(candidate)[0] == 1:
                        cf = candidate[0]
                        # at leaf level there is no remaining_min, all features fixed
                        min_dist = feature_dists.sum()

                # after evaluating the leaf, backtrack
                current_level -= 1

            else:
                # go one level deeper in the search tree
                current_level += 1



    # def exact_cf(self, x, target, dist_type=None, feature_order=None):
    #     """Exact branch-and-bound counterfactual search tailored to cautious random forests.

    #     Uses the EECE solution as an initial upper bound and then explores
    #     combinations of split intervals, with LOF-based plausibility filtering.

    #     feature_order:
    #         - if provided, use it directly
    #         - else: use np.arange(self.n_features)
    #     """
    #     if target not in self.classes:
    #         print("Your input target dose not existe!")
    #         return None, None

    #     if dist_type is None:
    #         dist_type = self.dist_type if self.dist_type is not None else 'L1'

    #     # decide which feature ordering to use
    #     if feature_order is None:
    #         feature_order = np.arange(self.n_features)
    #     else:
    #         feature_order = np.asarray(feature_order, dtype=int)

    #     start_time = time.time()
    #     # initial solution from OFCC (can be used as upper bound)
    #     init_cf, init_dist = self.ofcc(x, target, dist_type)
    #     if init_cf is None:
    #         init_cf, init_dist = self.discern(x, target, dist_type)
    #     if init_cf is None or init_dist is None:
    #         print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
    #         return None, None

    #     if dist_type == 'L2':
    #         init_dist = init_dist ** 2

    #     # filter split intervals according to current distance upper bound
    #     split_intervals, split_interval_dists, n_intervals = self.__filter_split_intervals(x, init_dist, dist_type)

    #     n_checked_intervals = np.zeros(self.n_features, dtype='int64')
    #     current_level = 0
    #     current_region = np.zeros(2 * self.n_features)
    #     feature_dists = np.zeros(self.n_features)
    #     leaf_index = {}

    #     cf = init_cf.copy()
    #     min_dist = float(init_dist)

    #     while True:
    #         end_time = time.time()
    #         if current_level < 0 or (end_time - start_time) > 8:
    #             # search finished or time budget exceeded
    #             if dist_type == 'L2':
    #                 min_dist = np.sqrt(min_dist)
    #             return cf, round(float(min_dist), 4)

    #         d = int(feature_order[current_level])

    #         # if no intervals or already checked all intervals for this feature -> backtrack
    #         if n_intervals[d] == 0 or n_checked_intervals[d] == n_intervals[d]:
    #             n_checked_intervals[d] = 0
    #             feature_dists[d] = 0.0
    #             current_level -= 1
    #             continue

    #         # take next interval for feature d
    #         feature_dists[d] = split_interval_dists[d][n_checked_intervals[d]]

    #         # branch-and-bound pruning
    #         if feature_dists.sum() >= min_dist:
    #             n_checked_intervals[d] = 0
    #             feature_dists[d] = 0.0
    #             current_level -= 1
    #             continue

    #         interval = split_intervals[d][int(n_checked_intervals[d])]
    #         interval_key = (interval[0], interval[1])
    #         n_checked_intervals[d] += 1
    #         current_region[2 * d:2 * d + 2] = interval

    #         # update leaf indices for this partial region
    #         if current_level == 0:
    #             leaf_index[current_level] = self.intervals_with_leaves[d][interval_key]
    #         else:
    #             leaf_index[current_level] = np.intersect1d(
    #                 leaf_index[current_level - 1],
    #                 self.intervals_with_leaves[d][interval_key]
    #             )

    #         # if we already fixed all features -> evaluate this region
    #         if current_level == self.n_features - 1:
    #             candidate = self.__generate_cf_in_regions(x, current_region.reshape((1, -1)))
    #             current_prediction = self.model.predict(candidate.reshape((1, -1)))[0]

    #             # check if we reach the target and enforce plausibility with LOF
    #             if current_prediction == target:
    #                 if self.lof.predict(candidate)[0] == 1:
    #                     cf = candidate[0]
    #                     min_dist = feature_dists.sum()

    #             # after exploring this leaf, backtrack
    #             current_level -= 1

    #         else:
    #             # go deeper in the search tree
    #             current_level += 1
    

    def generate_cf(self, x, target, generator='eece', dist_type=None, fi_type=None):
        """Generate a counterfactual using a chosen method and evaluate it.

        fi_type is only used when generator == 'exact_cf', to choose the feature_order:
        - fi_type is None:    feature_order = [0, 1, ..., p-1]
        - fi_type == 'global': feature_order = self.feature_order (from global FI in fit)
        - otherwise:          treated as a local FI method name
                              (e.g. 'PFI-Local', 'SHAP-Local', 'LIME-Local'),
                              and feature_order is np.argsort(local_importance)
        """
        x = np.asarray(x).ravel()
        y_hat = self.model.predict(x.reshape((1, -1)))[0]

        # prepare result container
        result = {
            'x': x,
            'y_hat': y_hat,
            'cf': None,
            'target': target,
            'valid': None,
            'dist_type': dist_type if dist_type is not None else self.dist_type,
            'L1': None,
            'L2': None,
            'L0': None,
            'plausible': None,
            'time cost': None}

        # basic target sanity check
        if target not in self.classes or target == y_hat:
            print("Your input target dose not existe!")
            return result

        # choose distance type for the generator
        if dist_type is None:
            dist_type = self.dist_type if self.dist_type is not None else 'L1'

        start_time = time.time()

        if generator == 'mo':
            cf, min_dist = self.mo(x, target, dist_type)
        elif generator == 'discern':
            cf, min_dist = self.discern(x, target, dist_type)
        elif generator == 'ofcc':
            cf, min_dist = self.ofcc(x, target, dist_type)
        elif generator == 'lire':
            cf, min_dist = self.lire(x, target, dist_type)
        elif generator == 'eece':
            cf, min_dist = self.eece(x, target, dist_type)
        elif generator == 'exact_cf':
            # fi_type to decide feature_order
            if fi_type is None:
                feature_order = None  # exact_cf will use range(n_features)
            elif fi_type == 'global':
                feature_order = self.feature_order
            else:
                # 'PFI-Local', 'SHAP-Local', 'LIME-Local'
                phi_local = self.compute_local_feature_importance(x, method=fi_type)
                feature_order = np.argsort(phi_local)
            
            start_time = time.time()
            cf, min_dist = self.exact_cf(x, target, dist_type, feature_order)
        else:
            # default to eece if unknown method name
            cf, min_dist = self.eece(x, target, dist_type)

        end_time = time.time()
        result['time cost'] = round(end_time - start_time, 5)

        if cf is None:
            return result

        result['cf'] = cf

        # prediction on the counterfactual
        cf_y_hat = self.model.predict(cf.reshape((1, -1)))[0]
        result['valid'] = (cf_y_hat == target)

        # distances
        result['L1'] = round(float(self.__instance_dist(x, cf.reshape((1, -1)), 'L1')[0]), 5)
        result['L2'] = round(float(self.__instance_dist(x, cf.reshape((1, -1)), 'L2')[0]), 5)
        result['L0'] = int(self.__instance_dist(x, cf.reshape((1, -1)), 'L0')[0])

        # plausibility via LOF
        try:
            plausibility = self.lof.predict(cf.reshape((1, -1)))[0]
            result['plausible'] = (plausibility == 1)
        except Exception:
            result['plausible'] = None

        return result
