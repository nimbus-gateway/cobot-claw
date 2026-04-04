from __future__ import annotations

from pathlib import Path

from cobot.cli.session import CobotSession
from cobot.cli.renderer import console
from cobot.scenarios.executor import ScenarioExecutor


async def cmd_run(session: CobotSession, name: str, dry_run: bool = False) -> None:
    if session.deps is None:
        console.print("[red]No robot connected.[/red]")
        return
    store = Path(session.cfg.scenarios.store_path)
    path = store / f"{name}.yaml"
    if not path.exists():
        console.print(f"[red]Scenario '{name}' not found.[/red]")
        return
    scenario = ScenarioExecutor.load(path)
    result = await session.deps.scenario_deps.executor.run(scenario, dry_run=dry_run)
    color = "green" if result.success else "red"
    console.print(f"[{color}]{result.steps_completed}/{result.total_steps} steps completed[/{color}]")
    if result.error:
        console.print(f"[red]{result.error}[/red]")


async def cmd_scenarios(session: CobotSession) -> None:
    store = Path(session.cfg.scenarios.store_path)
    names = [f.stem for f in store.glob("*.yaml")]
    if not names:
        console.print("[dim]No scenarios found.[/dim]")
        return
    for name in names:
        console.print(f"  [cyan]{name}[/cyan]")
