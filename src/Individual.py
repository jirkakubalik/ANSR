import ast
import inspect
import numpy as np
import copy
import os
import re
import torch

from typing import Any, List, Tuple, Dict

import SRConfig
from SRData import SRData
import SRUnits
from SubTopology import SubTopology


class PerformanceHistory:
    def __init__(
        self,
        performance,
        nn_weights,
        activeNodesCoordinates,
        dataTimestamp,
    ):
        self.performance: dict[str, float] = performance
        self.nn_weights: list[list[tuple[str, np.array]]] = nn_weights
        self.activeNodesCoordinates = activeNodesCoordinates
        self.dataTimestamp: float = dataTimestamp


class Individual:
    def __init__(
        self,
        inPopulation: str,
        indId: int,
        iters: int = 0,
        parents: tuple[str] = tuple(),
        # --- Genealogy is a dict[genealogy_tree_level: parents]
        genealogy: dict[int, tuple] = dict(),
    ):
        self.id = indId

        # --- Add a new record (<int>: parents) where <int> is 'highest_key-in_genealogy'+1
        self.parents = parents
        next_key = (max(genealogy.keys()) + 1) if genealogy else 0
        self.genealogy = {**genealogy, next_key: parents}

        self.inPopulation = inPopulation  # --- "mainPopulation", "offspring"
        self.subjectToTune = True

        self.age = iters
        self.backprop_iters = 0
        self.age_last_change_backprop = iters
        self.adult = getattr(self, SRConfig.ageToConsider) >= SRConfig.th_adult

        self.perf_history: dict[int, PerformanceHistory] = {}  # timestamp, history
        self.last_perf: PerformanceHistory = None
        self.last_output: np.array = np.array([])
        self.data_timestamp: float = 0.0
        self.front = SRConfig.maxFrontNb

        self.expr: str = ""
        self.simpleExpr: str = "NA"

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, Individual):
            return self.id == other.id
        return False

    def saveIndividualToFile(self, filePath: str):
        """ "
        Save the status of the individual to a text file.
        """

        with open(filePath, "w") as f:
            f.write(f"Id: {self.id}\n")
            f.write(f"Age: {self.age}\n")
            f.write(f"Age last change backprop: {self.age_last_change_backprop}\n")
            f.write(f"Backprop iterations: {self.backprop_iters}\n")
            f.write(f"Parents: {', '.join(map(str, self.parents))}\n")
            f.write(f"Genealogy: {repr(self.genealogy)}\n")
            f.write(f"Subject to tune: {self.subjectToTune}\n")
            f.write(f"Performance: {self.last_perf.performance}\n")
            saved_formula = self.expr if self.simpleExpr == "NA" else self.simpleExpr
            f.write(f"Formula: {saved_formula}")

            f.write("\n\nSubTopology node/weights:\n")

            classes = inspect.getmembers(SRUnits, inspect.isclass)

            nb_of_vars = {}
            for class_name, cls in classes:
                if getattr(cls, "nbOfVars", None):
                    nb_of_vars[class_name] = getattr(cls, "nbOfVars")

            for layer_idx, layer_params in enumerate(self.last_perf.nn_weights):
                node_idx = 0
                weight_idx = 0

                for unit_name, weight_value in layer_params:
                    if nb_of_vars.get(unit_name, 1) > weight_idx:
                        if isinstance(weight_value, torch.Tensor):
                            weight_value = weight_value.detach().cpu().tolist()
                        f.write(
                            f"name={unit_name}; layer={layer_idx}; node={node_idx}; weights={weight_idx}; value={weight_value}\n"
                        )
                        weight_idx += 1

                        if nb_of_vars.get(unit_name, 1) == weight_idx:
                            node_idx += 1
                            weight_idx = 0

            f.write(
                f"\nActive nodes coordinates: {self.last_perf.activeNodesCoordinates}\n"
            )

    @staticmethod
    def loadIndividualFromFile(filePath: str, ind_id: int):
        """
        Recreate an Individual from a text file produced by saveStatusOfIndividualToFile.
        Returns: Individual
        """
        layers: Dict[int, List[Tuple[int, int, str, Any]]] = {}
        in_weights_section = False

        active_nodes_coords = []

        with open(filePath, "r") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if line.strip() == "":
                    continue

                if line.strip().startswith("Active nodes coordinates:"):
                    in_weights_section = False
                    coordinates = line.split("Active nodes coordinates:", 1)[1].strip()
                    active_nodes_coords = ast.literal_eval(coordinates)
                    continue

                if line.strip().startswith("SubTopology node/weights:"):
                    in_weights_section = True
                    continue

                if in_weights_section:
                    m = re.match(
                        r"name=(.+); layer=(\d+); node=(\d+); weights=(\d+); value=(.+)",
                        line,
                    )
                    if not m:
                        continue
                    unit_name = m.group(1)
                    layer_idx = int(m.group(2))
                    node_idx = int(m.group(3))
                    weight_idx = int(m.group(4))
                    weight_value = ast.literal_eval(m.group(5))
                    weight_value = torch.tensor(weight_value, dtype=torch.float64)
                    layers.setdefault(layer_idx, []).append(
                        (node_idx, weight_idx, unit_name, weight_value)
                    )
                    continue

                if not in_weights_section:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        if key == "Performance":
                            perf_dict = ast.literal_eval(value)

        if layers:
            max_layer = max(layers.keys())
            nn_weights: List[List[Tuple[str, Any]]] = []
            for layer_idx in range(max_layer + 1):
                entries = layers.get(layer_idx, [])
                entries.sort(key=lambda t: (t[0], t[1]))
                layer_list: List[Tuple[str, Any]] = [
                    (name, value) for (_node, _widx, name, value) in entries
                ]
                nn_weights.append(layer_list)

        ind = Individual(
            inPopulation="mainPopulation", indId=ind_id, iters=0, parents=tuple()
        )
        ind.age_last_change_backprop = 0
        ind.backprop_iters = 0
        ind.subjectToTune = True
        ind.adult = getattr(ind, SRConfig.ageToConsider) >= SRConfig.th_adult

        subTopology = SubTopology.createSubtopologyFromParameters(
            nnWeights=nn_weights, activeNodesCoordinates=active_nodes_coords
        )

        _, nbOfImprovements = subTopology.train_nn(
            learningSteps=0,
            learningRate=SRConfig.learning_rate_newborn,
            regularize=True,
            clipWeights=True,
            deactivateBelowThresholdUnits=False,
        )

        performance = {
            "valid_loss": subTopology.losses["valid_loss"],
            "rmse_constr": subTopology.losses["rmse_constr"],
            "nbOfActiveNodes": subTopology.losses["nbOfActiveNodes"],
            "complexity": subTopology.losses["complexity"],
            "nbOfImprovements": nbOfImprovements,
        }
        ind.last_perf = PerformanceHistory(
            performance=performance,
            nn_weights=nn_weights,
            activeNodesCoordinates=active_nodes_coords,
            dataTimestamp=os.path.getmtime(SRConfig.train_data),
        )
        ind.perf_history[getattr(ind, SRConfig.ageToConsider)] = ind.last_perf
        ind.last_output = (
            subTopology.get_subtopology_output(SRData.x_data).detach().cpu().numpy()
        )

        return ind

    def saveIndividualPerformanceToFile(
        self, folderName: str, fileName: str, tickValue: str = "", prefix: str = "time_"
    ):
        """ "
        Save the performance metrics and analytic formulas of the individual to a text file.
        """
        completeFolderPath = (
            SRConfig.outputNamePrefix + folderName + f"/{prefix}{tickValue}/"
        )
        os.makedirs(
            completeFolderPath,
            exist_ok=True,
        )

        # --- Save Matlab file
        self.saveIndividualToMatlabFile(completeFolderPath, fileName)

        f = open(completeFolderPath + fileName + ".txt", "w")

        s = self.printPerformanceMetrics()
        f.write(s)
        # ---
        f.write(f"\n\nAge:{self.age}")
        f.write(f"\n\nBackprop iterations:{self.backprop_iters}")
        f.write(f"\n\nParents: {', '.join(map(str, self.parents))}")
        f.write(f"\n\nGenealogy: {repr(self.genealogy)}")
        # ---
        s = f"\n\nAnalytic formula in Python: {self.expr}\n"
        if self.simpleExpr:
            s += f"\n\nSimplified analytic formula: {self.simpleExpr}\n"
        # ---
        exprMatlab = (
            self.expr.replace("**", ".^")
            .replace("np.", "")
            .replace("*", ".*")
            .replace("/", "./")
        )
        s += f"\n\nAnalytic formula in MatLab: {exprMatlab}\n"
        if self.simpleExpr:
            simpleMatlab = (
                str(self.simpleExpr)
                .replace("**", ".^")
                .replace("np.", "")
                .replace("*", ".*")
                .replace("/", "./")
            )
            s += f"\n\nSimplified analytic formula in MatLab: {simpleMatlab}\n"

        f.write(s)
        f.close()

    def saveIndividualToMatlabFile(self, completeFolderPath: str, fileName: str):
        """
        Save the performance metrics and analytic formulas of the individual as Matlab function file.
        """
        # os.makedirs(
        #     SRConfig.outputNamePrefix + folderName + f"/time_{timestamp}/",
        #     exist_ok=True,
        # )
        # fileName = fileName.split(".")[0]
        f = open(completeFolderPath + fileName + ".m", "w")

        # --- Function header
        f.write(f"function output = {fileName}(input)\n")
        f.write(f"\n% Analytic model generated by openSR\n")

        # --- Performance metrics
        s = self.printPerformanceMetrics()
        prefixed = "\n".join("% " + line for line in s.splitlines())
        f.write(prefixed)
        # ---
        f.write(f"\n% Age: {self.age}")
        f.write(f"\n% Backprop iterations: {self.backprop_iters}")
        f.write(f"\n% Parents: {', '.join(map(str, self.parents))}\n")

        # --- Function body
        maxVars = SRData.x_data.shape[1]
        for i in range(maxVars):
            f.write(f"\nx{i+1} = input(:, {i+1});")

        exprMatlab = (
            self.expr.replace("**", ".^")
            .replace("np.", "")
            .replace("*", ".*")
            .replace("/", "./")
        )
        for i in reversed(range(maxVars)):
            old = rf"\bx{i}\b"
            new = f"x{i+1}"
            exprMatlab = re.sub(old, new, exprMatlab)
        f.write(f"\n\noutput = {exprMatlab};\n")
        f.close()

    def printPerformanceMetrics(self):
        s = f"Performance metrics:"
        s += f"\n\tnb. of active nodes: {int(self.last_perf.performance['nbOfActiveNodes'])}"
        s += f"\n\tcomplexity: {self.last_perf.performance['complexity']}"
        for key in self.last_perf.performance:
            if "rmse" in key:
                s += f"\n\t{key}: {self.last_perf.performance[key]:2.15f}"
            if "loss" in key:
                s += f"\n\t{key}: {self.last_perf.performance[key]:2.15f}"
        return s


def sortByRMSEValid(individual: Individual):
    return list(individual.perf_history.values())[-1].performance["valid_loss"]


def sortByRMSEConstr(individual: Individual):
    return list(individual.perf_history.values())[-1].performance["rmse_constr"]


def sortByNbOfActiveNodes(individual: Individual):
    return list(individual.perf_history.values())[-1].performance["nbOfActiveNodes"]


def sortByLoss(pop: list[Individual], lossName: str, reverse: bool = False):
    if lossName == "valid_loss":
        pop.sort(key=sortByRMSEValid, reverse=reverse)
    elif lossName == "rmse_constr":
        pop.sort(key=sortByRMSEConstr, reverse=reverse)
    elif lossName == "nbOfActiveNodes":
        pop.sort(key=sortByNbOfActiveNodes, reverse=reverse)
    return pop


def extractUniqueSolutionsBasedOnPerformance(
    population: list[Individual], performance_keys: list[str] = SubTopology.keysPerfCplx
):
    """
    Selects unique solutions based on performance metrics.
    """
    res = list(
        {
            tuple((k, ind.last_perf.performance[k]) for k in performance_keys): ind
            for ind in population
        }.values()
    )

    return res


def extractNondominatedSolutionsPerActiveNodes(
    population: list[Individual], archiveCounter: int
):
    """
    Selects nondominated solutions per every nbOfActiveNodes.
    """
    population = extractUniqueSolutionsBasedOnPerformance(
        population, performance_keys=SubTopology.keysPerfCplx
    )
    popDict: dict[int, list[Individual]] = (
        {}
    )  # --- nbOfActiveNodes -> list of individuals
    for ind in population:
        n = ind.last_perf.performance["nbOfActiveNodes"]
        popDict.setdefault(n, []).append(ind)
    res = set()
    for n in popDict.keys():
        for c in SubTopology.keysPerfCplx:
            popDict[n].sort(key=lambda ind: ind.last_perf.performance[c])
            res.add(copy.deepcopy(popDict[n][0]))
    res = list(res)

    for ind in res:
        if ind.id < SRConfig.archiveIdMin:
            ind.id = archiveCounter
            archiveCounter += 1

    return res, archiveCounter


def extractNondominatedSolutions(population: list[Individual], archiveCounter: int):
    """
    Extracts all nondominated solutions from the population.
    """
    criteria = SubTopology.keysPerfCplx
    population = extractUniqueSolutionsBasedOnPerformance(
        population, performance_keys=criteria
    )
    equalIdx: set[int] = set()
    front = 0

    res = set()
    for i in range(len(population)):
        if i in equalIdx:
            continue
        if (
            population[i].last_perf.performance["nbOfActiveNodes"]
            < SRConfig.minNbOfActiveNodes
        ):
            isDominated = True
        else:
            isDominated = False
            for j in range(len(population)):
                if i == j:
                    continue
                iDominates, jDominates, iEqualsj = getMutualDominance(
                    population[i].last_perf.performance,
                    population[j].last_perf.performance,
                    criteria=criteria,
                )
                if iEqualsj:
                    equalIdx.add(j)
                isDominated = isDominated or jDominates
        if not isDominated:
            population[i].front = front
            res.add(
                copy.deepcopy(population[i])
            )  # --- add i-th element to the list of non-dominated individuals
    res = list(res)

    for ind in res:
        if ind.id < SRConfig.archiveIdMin:
            ind.id = archiveCounter
            archiveCounter += 1

    return res, archiveCounter


def splitPopulationByDomination(pop: list[Individual], front: int, criteria: list[str]):
    nonDominatedSet: list[Individual] = []  # --- list of non-dominated individuals
    equalIdx: set[int] = set()
    dominatedIdx: set[int] = set()  # --- list of dominated individual indexes

    for i in range(len(pop)):
        if i in equalIdx:
            continue
        if (
            front == 0
            and list(pop[i].perf_history.values())[-1].performance["nbOfActiveNodes"]
            < SRConfig.minNbOfActiveNodes
        ):
            isDominated = True
        else:
            isDominated = False
            for j in range(len(pop)):
                if i == j:
                    continue
                iDominates, jDominates, iEqualsj = getMutualDominance(
                    list(pop[i].perf_history.values())[-1].performance,
                    list(pop[j].perf_history.values())[-1].performance,
                    criteria=criteria,
                )
                if iDominates:
                    dominatedIdx.add(j)  # --- add j into the list of dominated indexes
                if iEqualsj:
                    equalIdx.add(j)
                isDominated = isDominated or jDominates
        if not isDominated:
            pop[i].front = front
            # --- add i-th element to the list of non-dominated individuals
            nonDominatedSet.append(pop[i])
        else:
            dominatedIdx.add(i)
    dominatedSet = set([pop[el] for el in dominatedIdx if el not in equalIdx])

    filteredNonDominatedSet: set[Individual] = set()
    extremeSolutions = set()  # --- extreme solutions w.r.t. keysPerformance
    for k in SubTopology.keysPerformance:
        nonDominatedSet = sortByLoss(nonDominatedSet, k)
        usage = []
        if nonDominatedSet:
            extremeSolutions.add(nonDominatedSet[0])
            usage.append(
                list(nonDominatedSet[0].perf_history.values())[-1].performance[
                    "nbOfActiveNodes"
                ]
            )
        for sp in nonDominatedSet:
            if (
                list(sp.perf_history.values())[-1].performance["nbOfActiveNodes"]
                not in usage
            ):
                filteredNonDominatedSet.add(sp)
                usage.append(
                    list(sp.perf_history.values())[-1].performance["nbOfActiveNodes"]
                )
            elif SRConfig.enhancedDominated:
                dominatedSet.add(sp)
    filteredNonDominatedSet = [
        sp for sp in filteredNonDominatedSet if sp not in extremeSolutions
    ]
    filteredNonDominatedSet.sort(key=sortByNbOfActiveNodes)
    extremeSolutions = list(extremeSolutions)
    extremeSolutions.extend(filteredNonDominatedSet)
    filteredNonDominatedSet = extremeSolutions
    # ---
    # filteredNonDominatedSet = list(filteredNonDominatedSet)
    dominatedSet = [sp for sp in dominatedSet if sp not in filteredNonDominatedSet]
    # ---
    # filteredNonDominatedSet.sort(key=sortByNbOfActiveNodes)
    return filteredNonDominatedSet, dominatedSet


def nonDominatedSorting(
    population: list[Individual], finalSize: int, criteria: list[str]
):
    for ind in population:
        ind.front = SRConfig.maxFrontNb

    res = []
    dominatedSet = [ind for ind in population]
    front = 0
    while len(res) < finalSize and dominatedSet:
        nonDominatedSet, dominatedSet = splitPopulationByDomination(
            dominatedSet, front=front, criteria=criteria
        )
        res.extend(nonDominatedSet)
        front += 1
    res = res[:finalSize]  # --- clip to finalSize

    return res


def getMutualDominance(
    first: dict[str, float], second: dict[str, float], criteria: list[str]
):
    """
    Check mutual dominance of the two subTopologies w.r.t. the given criteria set.
    :param first: Performance metrics of the first subTopology.
    :param second: Performance metrics of the second subTopology.
    :param criteria: List of criteria for dominance comparison.
    :return: Tuple of (firstDominates, secondDominates, firstEqualsSecond).
    """
    firstDominates = False
    secondDominates = False
    firstEqualsSecond = False

    if len(criteria) == 1:
        key = criteria[0]
        if first[key] < second[key]:
            firstDominates = True
        elif first[key] > second[key]:
            secondDominates = True
        else:
            firstEqualsSecond = True
    else:
        firstIsBetter = any(first[key] < second[key] for key in criteria)
        secondIsBetter = any(first[key] > second[key] for key in criteria)

        if firstIsBetter and not secondIsBetter:
            firstDominates = True
        elif secondIsBetter and not firstIsBetter:
            secondDominates = True
        elif not firstIsBetter and not secondIsBetter:
            firstDominates = True
            firstEqualsSecond = True

    return firstDominates, secondDominates, firstEqualsSecond


def individualPerformanceSimilarity(
    ind1: Individual, indPop: list[Individual]
) -> list[float]:
    """
    Computes the performance similarity between subTopologies based on RMSE of their outputs.
    """

    similarity = []
    if not indPop:
        return similarity

    for ind2 in indPop:
        squaredDiff = (ind1.last_output - ind2.last_output) ** 2
        rmse = (squaredDiff.mean()) ** 0.5
        similarity.append(rmse)

    return similarity
