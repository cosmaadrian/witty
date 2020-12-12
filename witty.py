import argparse
from jinja2 import Template

parser = argparse.ArgumentParser(description='Do stuff.')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--create_model', type = str)
group.add_argument('--create_dataset',  type = str)
group.add_argument('--create_evaluator',  type = str)
group.add_argument('--create_trainer',  type = str)
parser.add_argument('--name', type = str, required = True)
args = parser.parse_args()


def update_nomenclature(args):
    import nomenclature

    dataset_names = [{'name': key, 'object_name': value.__name__} for key, value in nomenclature.DATASETS.items()]
    model_names = [{'name': key, 'object_name': value.__name__} for key, value in nomenclature.MODELS.items()]
    trainer_names = [{'name': key, 'object_name': value.__name__} for key, value in nomenclature.TRAINERS.items()]
    evaluator_names = [{'name': key, 'object_name': value.__name__} for key, value in nomenclature.EVALUATORS.items()]

    if args.create_model:
        model_names.append({
            'name': args.name,
            'object_name': args.create_model
        })

    if args.create_dataset:
        dataset_names.append({
            'name': args.name,
            'object_name': args.create_dataset
        })

    if args.create_trainer:
        trainer_names.append({
            'name': args.name,
            'object_name': args.create_trainer
        })

    if args.create_evaluator:
        evaluator_names.append({
            'name': args.name,
            'object_name': args.create_evaluator
        })

    with open('resource/templates/nomenclature.jinja', 'rt') as f:
        template = Template(f.read())
    output = template.render(model_names = model_names, trainer_names = trainer_names, dataset_names = dataset_names, evaluator_names = evaluator_names)

    with open('nomenclature.py', 'wt') as f:
        f.write(output)

def update_files(kind, name, object_name):
    with open(f'resource/templates/{kind}.jinja', 'rt') as f:
        template = Template(f.read())
    output = template.render(name = object_name)

    with open(f'{kind}/{name}.py', 'wt') as f:
        f.write(output)

    with open(f'{kind}/__init__.py', 'at') as f:
        f.write(f'from .{name} import {object_name}\n')

if args.create_model is not None:
    update_files(kind = 'models', name = args.name, object_name = args.create_model)

if args.create_dataset is not None:
    update_files(kind = 'datasets', name = args.name, object_name = args.create_dataset)

if args.create_trainer is not None:
    update_files(kind = 'trainers', name = args.name, object_name = args.create_trainer)

if args.create_evaluator is not None:
    update_files(kind = 'evaluators', name = args.name, object_name = args.create_evaluator)

update_nomenclature(args)

