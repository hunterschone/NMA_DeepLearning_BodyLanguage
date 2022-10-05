# AUTORIGHTS
# ---------------------------------------------------------
# Copyright (c) 2017, Saurabh Gupta
#
# This file is part of the VCOCO dataset hooks and is available
# under the terms of the Simplified BSD License provided in
# LICENSE. Please retain this notice and LICENSE if you use
# this file (or any portion of it) in your project.
# ---------------------------------------------------------

# vsrl_data is a dictionary for each action class:
# image_id       - Nx1
# ann_id         - Nx1
# label          - Nx1
# action_name    - string
# role_name      - ['agent', 'obj', 'instr']
# role_object_id - N x K matrix, obviously [:,0] is same as ann_id

import numpy as np
from coco_tools.pycocotools.coco import COCO
import os, json
import copy
import pickle
import pdb
import pandas as pd
import timeit

class VCOCOeval(object):

    def __init__(self, vsrl_annot_file, coco_annot_file,
                 split_file):
        """Input:
    vslr_annot_file: path to the vcoco annotations
    coco_annot_file: path to the coco annotations
    split_file: image ids for split
    """
        self.COCO = COCO(coco_annot_file)
        self.VCOCO = _load_vcoco(vsrl_annot_file)
        # self.image_ids = np.loadtxt(open(split_file, 'r'))
        self.image_ids = open(split_file, 'r').read().split('\n')[:-1]
        # simple check
        # assert np.all(np.equal(np.sort(np.unique(self.VCOCO[0]['image_id'])), self.image_ids))

        self._init_coco()
        self._init_vcoco()

        self.agent_verb_list = []
        print('---------Setting agent_verb_list------------------')
        for aid in range(self.num_actions):
            self.agent_verb_list.append(self.actions[aid])

        self.role_verb_list = []
        print('---------Setting role_verb_list------------------')
        for aid in range(self.num_actions):
            for rid in range(len(self.roles[aid]) - 1):
                self.role_verb_list.append(self.actions[aid] + '-' + self.roles[aid][rid + 1])


    def _init_vcoco(self):
        actions = [x['action_name'] for x in self.VCOCO]
        roles = [x['role_name'] for x in self.VCOCO]
        self.actions = actions
        self.actions_to_id_map = {v: i for i, v in enumerate(self.actions)}
        self.num_actions = len(self.actions)
        self.roles = roles

    def _init_coco(self):
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}

    def _get_vcocodb(self):
        # self.image_ids = self.image_ids.astype(int)
        vcocodb = copy.deepcopy(self.COCO.loadImgs(self.image_ids))
        for entry in vcocodb:
            self._prep_vcocodb_entry(entry)
            self._add_gt_annotations(entry)

        # print
        if 0:
            nums = np.zeros((self.num_actions), dtype=np.int32)
            for entry in vcocodb:
                for aid in range(self.num_actions):
                    nums[aid] += np.sum(np.logical_and(entry['gt_actions'][:, aid] == 1, entry['gt_classes'] == 1))
            for aid in range(self.num_actions):
                print('Action %s = %d' % (self.actions[aid], nums[aid]))

        return vcocodb

    def _prep_vcocodb_entry(self, entry):
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['gt_actions'] = np.empty((0, self.num_actions), dtype=np.int32)
        entry['gt_role_id'] = np.empty((0, self.num_actions, 2), dtype=np.int32)

    def _add_gt_annotations(self, entry):
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_ann_ids = []
        width = entry['width']
        height = entry['height']
        for i, obj in enumerate(objs):
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = obj['bbox'][0]
            y1 = obj['bbox'][1]
            x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)
            x1, y1, x2, y2 = clip_xyxy_to_image(
                x1, y1, x2, y2, height, width)
            # Require non-zero seg area and more than 1x1 box size
            # if obj['area'] > 0 and x2 > x1 and y2 > y1:
            #     obj['clean_bbox'] = [x1, y1, x2, y2]
            #     valid_objs.append(obj)
            #     valid_ann_ids.append(ann_ids[i])
            obj['clean_bbox'] = [x1, y1, x2, y2]
            valid_objs.append(obj)
            valid_ann_ids.append(ann_ids[i])
        num_valid_objs = len(valid_objs)
        assert num_valid_objs == len(valid_ann_ids)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_actions = -np.ones((num_valid_objs, self.num_actions), dtype=entry['gt_actions'].dtype)
        gt_role_id = -np.ones((num_valid_objs, self.num_actions, 2), dtype=object)

        for ix, obj in enumerate(valid_objs):
            # cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            cls = obj['category_id']
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            is_crowd[ix] = obj['iscrowd']

            gt_actions[ix, :], gt_role_id[ix, :, :] = \
                self._get_vsrl_data(valid_ann_ids[ix],
                                    valid_ann_ids, valid_objs)

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['gt_actions'] = np.append(entry['gt_actions'], gt_actions, axis=0)
        entry['gt_role_id'] = np.append(entry['gt_role_id'], gt_role_id, axis=0)

    def _get_vsrl_data(self, ann_id, ann_ids, objs):
        """ Get VSRL data for ann_id."""
        action_id = -np.ones((self.num_actions), dtype=np.int32)
        role_id = -np.ones((self.num_actions, 2), dtype=object)
        # check if ann_id in vcoco annotations
        in_vcoco = np.where(self.VCOCO[0]['ann_id'] == ann_id)[0]
        if in_vcoco.size > 0:
            action_id[:] = 0
            role_id[:] = -1
        else:
            return action_id, role_id
        for i, x in enumerate(self.VCOCO):
            assert x['action_name'] == self.actions[i]
            has_label = np.where(np.logical_and(x['ann_id'] == ann_id, x['label'] == 1))[0]
            if has_label.size > 0:
                action_id[i] = 1
                # assert has_label.size == 1
                rids = x['role_object_id'][has_label]
                assert rids[0, 0] == ann_id
                '''
                for j in range(1, rids.shape[1]):
                if rids[0, j] == 0:
                    # no role
                    continue
                aid = np.where(ann_ids == rids[0, j])[0]
                assert aid.size > 0
                role_id[i, j - 1] = aid
                '''
                if has_label.size > 1:
                    aid = []
                    for k in range(rids.shape[0]):
                        if rids[k, 1] == 0:
                            # no role
                            continue
                        aid.append(int(np.where(ann_ids == rids[k, 1])[0]))
                    role_id[i, 0] = aid
                else:
                    for j in range(1, rids.shape[1]):
                        if rids[0, j] == 0:
                            # no role
                            continue
                        aid = np.where(ann_ids == rids[0, j])[0]
                        assert aid.size > 0
                        role_id[i, j - 1] = aid

        return action_id, role_id

    def _collect_detections_for_image(self, dets, image_id):
        agents = np.empty((0, 4 + self.num_actions), dtype=np.float32)
        roles = np.empty((0, 5 * self.num_actions, 2), dtype=np.float32)
        associations = np.empty((0), dtype=np.int32)

        agent_idx = 0
        for det in dets:
            if det['image_id'] == image_id:
                this_agent = np.zeros((1, 4 + self.num_actions), dtype=np.float32)
                this_agent[0, :4] = det['person_box']

                # Search for maximum number of role
                max_role = 0
                for aid in range(self.num_actions):
                    for j, rid in enumerate(self.roles[aid]):
                        if rid != 'agent':
                            max_role = max(max_role, len(det[self.actions[aid] + '_' + rid]))
                this_role = -np.ones((max_role, 5 * self.num_actions, 2), dtype=np.float32)
                this_association = np.full((max_role), agent_idx)
                agent_idx += 1

                for aid in range(self.num_actions):
                    for j, rid in enumerate(self.roles[aid]):
                        if rid == 'agent':
                            this_agent[0, 4 + aid] = det[self.actions[aid] + '_' + rid]
                        else:
                            '''
                            this_role[0, 5 * aid: 5 * aid + 5, j-1] = det[self.actions[aid] + '_' + rid]
                            '''
                            role_temp = det[self.actions[aid] + '_' + rid]
                            for i in range(len(role_temp)):
                                this_role[i, 5 * aid: 5 * aid + 5, j - 1] = role_temp[i]

                agents = np.concatenate((agents, this_agent), axis=0)
                roles = np.concatenate((roles, this_role), axis=0)
                associations = np.concatenate((associations, this_association), axis=0)
        return agents, roles, associations

    def _do_eval(self, detections_file, image_evaluated_id, ovr_thresh=0.5):
        self.image_ids = image_evaluated_id
        vcocodb = self._get_vcocodb()
        metrics = []

        # # HACK
        # detections_file = detections_file[:-4] + "_gt.pkl"
        # image_evaluated_id = [ann['id'] for ann in vcocodb]
        # print(len(image_evaluated_id))

        m_agent, results_per_verb_agent = self._do_agent_eval(vcocodb, detections_file, image_evaluated_id, ovr_thresh=ovr_thresh)
        m_role1, results_per_verb_role1 = self._do_role_eval(vcocodb, detections_file, image_evaluated_id, ovr_thresh=ovr_thresh,
                                          eval_type='scenario_1')
        m_role2, results_per_verb_role2 = self._do_role_eval(vcocodb, detections_file, image_evaluated_id, ovr_thresh=ovr_thresh,
                                          eval_type='scenario_2')

        metrics.append(m_agent)
        metrics.append(m_role1)
        metrics.append(m_role2)

        return metrics, results_per_verb_agent, results_per_verb_role1, results_per_verb_role2

    def _do_role_eval(self, vcocodb, detections_file, image_evaluated_id, ovr_thresh=0.5, eval_type='scenario_1'):

        with open(detections_file, 'rb') as f:
            dets = pickle.load(f)
        tp = [[[] for r in range(2)] for a in range(self.num_actions)]
        fp = [[[] for r in range(2)] for a in range(self.num_actions)]
        sc = [[[] for r in range(2)] for a in range(self.num_actions)]

        # npos = np.zeros((self.num_actions), dtype=np.float32)
        npos = []
        for aid in range(self.num_actions):
            npos_for_verb = [0]
            if len(self.roles[aid]) >= 2:
                npos_for_verb = [0] * (len(self.roles[aid]) - 1)

            npos.append(npos_for_verb)

        # debug_aid = 16
        # tp_prev_size = [0]
        # npos_prev = [0]
        # if len(self.roles[debug_aid]) >= 2:
        #     npos_prev = [0] * (len(self.roles[debug_aid]) - 1)
        #     tp_prev_size = [0] * (len(self.roles[debug_aid]) - 1)

        for i in range(len(vcocodb)):
            image_id = vcocodb[i]['id']
            if image_id in image_evaluated_id:
                gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]
                # person boxes
                gt_boxes = vcocodb[i]['boxes'][gt_inds]
                gt_actions = vcocodb[i]['gt_actions'][gt_inds]
                # some peorson instances don't have annotated actions
                # we ignore those instances
                ignore = np.any(gt_actions == -1, axis=1)
                assert np.all(gt_actions[np.where(ignore == True)[0]] == -1)


                pred_agents, pred_roles, association_idx = self._collect_detections_for_image(dets, image_id)

                for aid in range(self.num_actions):
                    if len(self.roles[aid]) < 2:
                        # if action has no role, then no role AP computed
                        continue

                    for rid in range(len(self.roles[aid]) - 1):
                        # keep track of detected instances for each action for each role
                        covered = np.zeros((gt_boxes.shape[0], vcocodb[i]['boxes'].shape[0] + 1), dtype=np.bool)
                        # get gt roles for action and role
                        gt_role_inds = vcocodb[i]['gt_role_id'][gt_inds, aid, rid]

                        total_num_role = 0
                        for role_person in gt_role_inds:
                            if isinstance(role_person, int):
                                num_role = 0
                            else:
                                num_role = len(role_person)
                            total_num_role += num_role

                        gt_roles = -np.ones((total_num_role, 4))
                        gt_association = -np.ones((total_num_role))
                        j = 0
                        for idx_person, role_person in enumerate(gt_role_inds):
                            if not isinstance(role_person, int):
                                for r in role_person:
                                    gt_roles[j] = vcocodb[i]['boxes'][r]
                                    gt_association[j] = idx_person
                                    j += 1

                        npos_for_verb = 0
                        for val_idx, val in enumerate(gt_actions[:, aid]):
                            if val == 1:
                                sum = np.sum(gt_association == val_idx)
                                if sum > 0:
                                    npos_for_verb += sum
                                else:
                                    npos_for_verb += 1
                        npos[aid][rid] += npos_for_verb

                        agent_boxes = pred_agents[:, :4]
                        role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4, rid]
                        agent_scores = pred_roles[:, 5 * aid + 4, rid]

                        # np.isnan does not accept obj types
                        valid = np.where(np.logical_and(np.logical_not(np.isnan(agent_scores)), np.not_equal(agent_scores, -1)))[0]

                        agent_scores = agent_scores[valid]
                        role_boxes = role_boxes[valid, :]
                        valid_association_idx = association_idx[valid]

                        # At least one agent box is associated to one role box
                        if valid_association_idx.shape[0] > 0:
                            # idx_to_keep = np.array(np.ediff1d(np.hstack((valid_association_idx[0] - 1, valid_association_idx))), dtype=bool)
                            # unique_association_idx = valid_association_idx[idx_to_keep]
                            # agent_boxes = agent_boxes[unique_association_idx, :]

                            idx = agent_scores.argsort()[::-1]

                            for j in idx:
                                j_person = valid_association_idx[j]
                                pred_box = agent_boxes[j_person, :]
                                overlaps = get_overlap(gt_boxes, pred_box)

                                # matching happens based on the person
                                jmax = overlaps.argmax()
                                ovmax = overlaps.max()

                                # if matched with an instance with no annotations
                                # continue
                                if ignore[jmax]:
                                    continue

                                # overlap between predicted role and gt role
                                associated_idx = gt_association == jmax
                                if associated_idx.shape[0] != 0 and np.any(associated_idx):
                                    # Several role box for the agent box. Search for the best overlapping
                                    gt_role_for_associated_person = gt_roles[associated_idx, :]

                                    ov_roles = []
                                    for t, target_box in enumerate(gt_role_for_associated_person):
                                        ov_roles.append(get_overlap(target_box.reshape((1, 4)), role_boxes[j, :]))

                                    ov_role_max = np.array(ov_roles).argmax()
                                    ov_role = ov_roles[ov_role_max]
                                else:
                                    # No role boxes for the agent. Look for scenarii
                                    ov_role_max = -1
                                    if eval_type == 'scenario_1':
                                        if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                                            # if no role is predicted, mark it as correct role overlap
                                            ov_role = 1.0
                                        else:
                                            # if a role is predicted, mark it as false
                                            ov_role = 0.0
                                    elif eval_type == 'scenario_2':
                                        # if no gt role, role prediction is always correct, irrespective of the actual predition
                                        ov_role = 1.0
                                    else:
                                        raise ValueError('Unknown eval type')

                                is_true_action = (gt_actions[jmax, aid] == 1)
                                sc[aid][rid].append(agent_scores[j])
                                if is_true_action and (ovmax >= ovr_thresh) and (ov_role >= ovr_thresh):
                                    if covered[jmax][ov_role_max]:
                                        fp[aid][rid].append(1)
                                        tp[aid][rid].append(0)
                                    else:
                                        fp[aid][rid].append(0)
                                        tp[aid][rid].append(1)
                                        covered[jmax][ov_role_max] = True
                                else:
                                    fp[aid][rid].append(1)
                                    tp[aid][rid].append(0)
                        else:
                            # Special case no role box set a null score to not considering it
                            sc[aid][rid].append(0.0)
                            fp[aid][rid].append(1)
                            tp[aid][rid].append(0)


                # if len(self.roles[debug_aid]) < 2:
                #     # if action has no role, then no role AP computed
                #     continue
                #
                # for rid in range(len(self.roles[debug_aid]) - 1):
                #     if npos[debug_aid][rid] != npos_prev[rid]:
                #         sum_tp = 0
                #         gt_npos = npos[debug_aid][rid] - npos_prev[rid]
                #         # print(self.actions[debug_aid], self.roles[debug_aid])
                #         current_tp = tp[debug_aid][rid][tp_prev_size[rid]:]
                #         current_sc = sc[debug_aid][rid][tp_prev_size[rid]:]
                #         for true_pos, det_sc in zip(current_tp, current_sc):
                #             sum_tp += true_pos
                #             if true_pos != 1 and det_sc > 0.5:
                #                 print('Wrong hight score !!! pb on', image_id, det_sc)
                #             elif true_pos == 1 and det_sc <= 0.5:
                #                 print('Wrong low score !!! pb on', image_id, det_sc)
                #
                #         if sum_tp != gt_npos:
                #             print('Wrong detection for rid', rid, tp_prev_size[rid], len(tp[debug_aid][rid]), current_tp)
                #             print('pb on', image_id, sum_tp, gt_npos)
                #     npos_prev[rid] = npos[debug_aid][rid]
                #     tp_prev_size[rid] = len(tp[debug_aid][rid])

        # compute ap for each action
        role_ap = np.zeros((self.num_actions, 2), dtype=np.float32)
        role_ap[:] = np.nan
        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2:
                continue
            for rid in range(len(self.roles[aid]) - 1):
                a_fp = np.array(fp[aid][rid], dtype=np.float32)
                a_tp = np.array(tp[aid][rid], dtype=np.float32)
                a_sc = np.array(sc[aid][rid], dtype=np.float32)

                # sort in descending score order
                idx = a_sc.argsort()[::-1]
                a_fp = a_fp[idx]
                a_tp = a_tp[idx]
                # a_sc = a_sc[idx]

                a_fp = np.cumsum(a_fp)
                a_tp = np.cumsum(a_tp)

                if npos[aid][rid] == 0:
                    rec = a_tp
                else:
                    rec = a_tp / float(npos[aid][rid])

                # check
                assert (np.amax(rec) <= 1)
                prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
                role_ap[aid, rid] = voc_ap(rec, prec)

        results_per_verb = []
        print('---------Reporting Role AP (%)------------------')
        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2: continue
            for rid in range(len(self.roles[aid]) - 1):
                print('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid] + '-' + self.roles[aid][rid + 1],
                                                                   role_ap[aid, rid] * 100.0, int(npos[aid][rid])))
                results_per_verb.append(role_ap[aid, rid] * 100.0)
        print('Average Role [%s] AP = %.2f' % (eval_type, np.nanmean(role_ap) * 100.00))
        print('---------------------------------------------')

        return np.nanmean(role_ap) * 100.00, results_per_verb

    def _do_agent_eval(self, vcocodb, detections_file, image_evaluated_id, ovr_thresh=0.5):

        with open(detections_file, 'rb') as f:
            dets = pickle.load(f)

        tp = [[] for a in range(self.num_actions)]
        fp = [[] for a in range(self.num_actions)]
        sc = [[] for a in range(self.num_actions)]

        npos = np.zeros((self.num_actions), dtype=np.float32)

        # print('init size tp', len(tp), len(tp[0]))
        # tp_prev_size = 0
        # npos_prev = 0

        for i in range(len(vcocodb)):
            image_id = vcocodb[i]['id']
            if image_id in image_evaluated_id:
                gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]
                # person boxes
                gt_boxes = vcocodb[i]['boxes'][gt_inds]
                gt_actions = vcocodb[i]['gt_actions'][gt_inds]
                # some person instances don't have annotated actions
                # we ignore those instances
                ignore = np.any(gt_actions == -1, axis=1)
                for aid in range(self.num_actions):
                    npos[aid] += np.sum(gt_actions[:, aid] == 1)

                pred_agents, _, _ = self._collect_detections_for_image(dets, image_id)
                for aid in range(self.num_actions):

                    # keep track of detected instances for each action
                    covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)

                    agent_scores = pred_agents[:, 4 + aid]
                    agent_boxes = pred_agents[:, :4]
                    # remove NaNs
                    valid = np.where(pd.isnull(agent_scores) == False)[0]
                    agent_scores = agent_scores[valid]
                    agent_boxes = agent_boxes[valid, :]

                    # sort in descending order
                    idx = agent_scores.argsort()[::-1]

                    for j in idx:
                        pred_box = agent_boxes[j, :]
                        overlaps = get_overlap(gt_boxes, pred_box)

                        jmax = overlaps.argmax()
                        ovmax = overlaps.max()

                        # if matched with an instance with no annotations
                        # continue
                        if ignore[jmax]:
                            continue

                        is_true_action = (gt_actions[jmax, aid] == 1)
                        sc[aid].append(agent_scores[j])
                        if is_true_action and (ovmax >= ovr_thresh):
                            if covered[jmax]:
                                fp[aid].append(1)
                                tp[aid].append(0)
                            else:
                                fp[aid].append(0)
                                tp[aid].append(1)
                                covered[jmax] = True
                        else:
                            fp[aid].append(1)
                            tp[aid].append(0)

                # debug_aid = 15
                # if npos[debug_aid] != npos_prev:
                #    sum_tp = 0
                #    gt_npos = npos[debug_aid] - npos_prev
                #    # print(self.actions[debug_aid], self.roles[debug_aid])
                #    current_tp = tp[debug_aid][tp_prev_size:]
                #    for true_pos in current_tp:
                #        sum_tp += true_pos
                #    if sum_tp != gt_npos:
                #        print('####', tp_prev_size, len(tp[debug_aid]), current_tp)
                #        print('pb on', image_id, sum_tp, gt_npos)
                #
                # npos_prev = npos[debug_aid]
                # tp_prev_size = len(tp[debug_aid])

        # compute ap for each action
        agent_ap = np.zeros((self.num_actions), dtype=np.float32)
        for aid in range(self.num_actions):

            a_fp = np.array(fp[aid], dtype=np.float32)
            a_tp = np.array(tp[aid], dtype=np.float32)
            a_sc = np.array(sc[aid], dtype=np.float32)
            # sort in descending score order
            idx = a_sc.argsort()[::-1]
            a_fp = a_fp[idx]
            a_tp = a_tp[idx]
            # a_sc = a_sc[idx]

            a_fp = np.cumsum(a_fp)
            a_tp = np.cumsum(a_tp)

            if npos[aid] == 0:
                rec = a_tp
            else:
                rec = a_tp / float(npos[aid])
            # check
            assert (np.amax(rec) <= 1)
            prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
            agent_ap[aid] = voc_ap(rec, prec)

        results_per_verb = []
        print('---------Reporting Agent AP (%)------------------')
        for aid in range(self.num_actions):
            print(
                '{: >20}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid], agent_ap[aid] * 100.0, int(npos[aid])))
            results_per_verb.append(agent_ap[aid] * 100.0)
        print('Average Agent AP = %.2f' % (np.nansum(agent_ap) * 100.00 / self.num_actions))
        print('---------------------------------------------')

        return np.nansum(agent_ap) * 100.00 / self.num_actions, results_per_verb


def _load_vcoco(vcoco_file):
    print('loading vcoco annotations...')
    with open(vcoco_file, 'r') as f:
        vsrl_data = json.load(f)
    for i in range(len(vsrl_data)):
        vsrl_data[i]['role_object_id'] = \
            np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']), -1)).T
        for j in ['ann_id', 'label', 'image_id']:
            vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1, 1))
    return vsrl_data

def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2

def get_overlap(boxes, ref_box):
    ixmin = np.maximum(boxes[:, 0], ref_box[0])
    iymin = np.maximum(boxes[:, 1], ref_box[1])
    ixmax = np.minimum(boxes[:, 2], ref_box[2])
    iymax = np.minimum(boxes[:, 3], ref_box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) +
           (boxes[:, 2] - boxes[:, 0] + 1.) *
           (boxes[:, 3] - boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec)
    Compute VOC AP given precision and recall.
    [as defined in PASCAL VOC]
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
