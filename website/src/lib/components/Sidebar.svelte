<script lang="ts">
	import { page } from '$app/stores';
	import { indicatorCategories } from '$lib/stores';
	
	let isOpen = false;
	
	function toggleSidebar() {
		isOpen = !isOpen;
	}
	
	function closeSidebar() {
		isOpen = false;
	}
	
	$: categories = $indicatorCategories;
</script>

<!-- Mobile menu toggle -->
<button class="menu-toggle" on:click={toggleSidebar} aria-label="Toggle navigation menu">
	<span class="hamburger" class:open={isOpen}>
		<span></span>
		<span></span>
		<span></span>
	</span>
</button>

<!-- Sidebar -->
<aside class="sidebar" class:open={isOpen}>
	<div class="logo">
		<div class="logo-icon">
			<svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
				<path d="M3 3L21 21M9 9L21 3L15 15L9 9Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
				<path d="M12 12L3 21L9 15L12 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
			</svg>
		</div>
		<div class="logo-text">
			<h2>Rust-Backtester</h2>
			<p>Technical Analysis</p>
		</div>
	</div>
	
	<nav class="nav-menu">
		<div class="nav-section">
			<h3>
				<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
					<path d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
				</svg>
				Overview
			</h3>
			<ul>
				<li>
					<a 
						href="/" 
						class:active={$page.url.pathname === '/'} 
						on:click={closeSidebar}
					>
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
							<path d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
						</svg>
						Home
					</a>
				</li>
			</ul>
		</div>

		{#each Object.entries(categories) as [categoryKey, indicators]}
			<div class="nav-section">
				<h3>
					{#if categoryKey === 'moving-averages'}
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
							<path d="M21 6H3M21 12H9M21 18H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
						</svg>
						Moving Averages
					{:else if categoryKey === 'momentum'}
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
							<path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
						</svg>
						Momentum
					{:else if categoryKey === 'volatility'}
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
							<path d="M8 18L12 6L16 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
							<path d="M9.5 13.5H14.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
						</svg>
						Volatility
					{:else if categoryKey === 'volume'}
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
							<path d="M11 5L6 9L2 9V15L6 15L11 19V5Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
							<path d="M19.07 4.93C20.9447 6.80528 21.9979 9.34836 21.9979 12C21.9979 14.6516 20.9447 17.1947 19.07 19.07" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
							<path d="M15.54 8.46C16.4774 9.39764 17.0039 10.6692 17.0039 12C17.0039 13.3308 16.4774 14.6024 15.54 15.54" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
						</svg>
						Volume
					{:else if categoryKey === 'trend'}
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
							<path d="M22 12L18 8L14 12L10 8L6 12L2 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
						</svg>
						Trend
					{:else}
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
							<path d="M9 19C9 20.1046 7.65685 21 6 21C4.34315 21 3 20.1046 3 19C3 17.8954 4.34315 17 6 17C7.65685 17 9 17.8954 9 19Z" stroke="currentColor" stroke-width="2"/>
							<path d="M21 8C21 9.10457 19.6569 10 18 10C16.3431 10 15 9.10457 15 8C15 6.89543 16.3431 6 18 6C19.6569 6 21 6.89543 21 8Z" stroke="currentColor" stroke-width="2"/>
							<path d="M15 14L9 14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
							<path d="M8 14L21 3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
						</svg>
						{categoryKey}
					{/if}
				</h3>
				<ul>
					{#each indicators as indicator}
						<li>
							<a 
								href="/indicators/{indicator.id}" 
								class:active={$page.url.pathname === `/indicators/${indicator.id}`}
								on:click={closeSidebar}
							>
								{indicator.name}
								<span class="indicator-abbr">{indicator.id.toUpperCase()}</span>
							</a>
						</li>
					{/each}
				</ul>
			</div>
		{/each}
	</nav>
</aside>

<!-- Overlay for mobile -->
{#if isOpen}
	<div 
		class="overlay" 
		role="button" 
		tabindex="0"
		on:click={closeSidebar}
		on:keydown={(e) => e.key === 'Escape' && closeSidebar()}
		aria-label="Close navigation menu"
	></div>
{/if}

<style>
	.menu-toggle {
		display: none;
		position: fixed;
		top: var(--space-4);
		left: var(--space-4);
		z-index: 1001;
		background: var(--primary-color);
		color: white;
		border: none;
		border-radius: var(--radius);
		padding: var(--space-3);
		cursor: pointer;
		width: 48px;
		height: 48px;
		box-shadow: var(--shadow-lg);
		transition: var(--transition);
	}

	.menu-toggle:hover {
		background: var(--primary-dark);
		transform: scale(1.05);
	}

	.hamburger {
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		width: 20px;
		height: 16px;
		transition: var(--transition);
	}

	.hamburger span {
		display: block;
		height: 2px;
		background: currentColor;
		border-radius: 1px;
		transition: var(--transition);
	}

	.hamburger.open span:nth-child(1) {
		transform: translateY(7px) rotate(45deg);
	}

	.hamburger.open span:nth-child(2) {
		opacity: 0;
	}

	.hamburger.open span:nth-child(3) {
		transform: translateY(-7px) rotate(-45deg);
	}

	.sidebar {
		width: 280px;
		background: var(--surface);
		border-right: 1px solid var(--border);
		position: fixed;
		height: 100vh;
		overflow-y: auto;
		padding: 0;
		z-index: 100;
		transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
		box-shadow: var(--shadow-lg);
	}

	.logo {
		padding: var(--space-6);
		border-bottom: 1px solid var(--border);
		background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
		color: white;
		display: flex;
		align-items: center;
		gap: var(--space-3);
		position: relative;
		overflow: hidden;
	}

	.logo::before {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M 8 0 L 0 0 0 8" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
		pointer-events: none;
	}

	.logo-icon {
		position: relative;
		z-index: 1;
		flex-shrink: 0;
		width: 32px;
		height: 32px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(255, 255, 255, 0.1);
		border-radius: var(--radius);
		backdrop-filter: blur(10px);
	}

	.logo-text {
		position: relative;
		z-index: 1;
	}

	.logo h2 {
		font-size: 1.125rem;
		font-weight: 700;
		margin: 0 0 var(--space-1) 0;
		color: white;
	}

	.logo p {
		font-size: 0.8rem;
		opacity: 0.9;
		margin: 0;
		color: white;
		font-weight: 400;
	}

	.nav-menu {
		padding: var(--space-4) 0;
	}

	.nav-section {
		margin-bottom: var(--space-6);
	}

	.nav-section h3 {
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
		padding: 0 var(--space-6);
		margin-bottom: var(--space-3);
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}

	.nav-section h3 svg {
		width: 14px;
		height: 14px;
		opacity: 0.7;
	}

	.nav-section ul {
		list-style: none;
		margin: 0;
		padding: 0;
	}

	.nav-section ul li {
		margin: 0;
	}

	.nav-section ul li a {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--space-3) var(--space-6);
		color: var(--text-primary);
		text-decoration: none;
		font-size: 0.875rem;
		transition: var(--transition);
		border-left: 3px solid transparent;
		position: relative;
		font-weight: 400;
	}

	.nav-section ul li a::before {
		content: '';
		position: absolute;
		left: 0;
		top: 0;
		bottom: 0;
		width: 3px;
		background: var(--primary-color);
		transform: scaleY(0);
		transition: transform 0.2s ease;
	}

	.nav-section ul li a:hover {
		background-color: var(--surface-elevated);
		color: var(--primary-color);
		padding-left: var(--space-8);
	}

	.nav-section ul li a:hover::before {
		transform: scaleY(1);
	}

	.nav-section ul li a.active {
		background: linear-gradient(90deg, var(--primary-color), var(--primary-light));
		color: white;
		border-left-color: var(--accent-color);
		font-weight: 500;
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
	}

	.nav-section ul li a.active::before {
		transform: scaleY(1);
		background: var(--accent-color);
	}

	.nav-section ul li a svg {
		width: 16px;
		height: 16px;
		margin-right: var(--space-2);
		opacity: 0.7;
	}

	.indicator-abbr {
		font-size: 0.7rem;
		font-weight: 600;
		opacity: 0.6;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		background: rgba(0, 0, 0, 0.05);
		padding: var(--space-1) var(--space-2);
		border-radius: var(--radius-sm);
		transition: var(--transition);
	}

	.nav-section ul li a:hover .indicator-abbr,
	.nav-section ul li a.active .indicator-abbr {
		background: rgba(255, 255, 255, 0.2);
		opacity: 1;
	}

	.overlay {
		display: none;
		position: fixed;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		background: rgba(0, 0, 0, 0.6);
		z-index: 99;
		backdrop-filter: blur(4px);
	}

	/* Custom scrollbar */
	.sidebar::-webkit-scrollbar {
		width: 6px;
	}

	.sidebar::-webkit-scrollbar-track {
		background: var(--surface-elevated);
	}

	.sidebar::-webkit-scrollbar-thumb {
		background: var(--border);
		border-radius: 3px;
	}

	.sidebar::-webkit-scrollbar-thumb:hover {
		background: var(--text-tertiary);
	}

	@media (max-width: 1024px) {
		.menu-toggle {
			display: flex;
			align-items: center;
			justify-content: center;
		}

		.sidebar {
			transform: translateX(-100%);
		}

		.sidebar.open {
			transform: translateX(0);
		}

		.overlay {
			display: block;
		}
	}

	@media (max-width: 480px) {
		.sidebar {
			width: 100vw;
		}

		.logo {
			padding: var(--space-4);
		}

		.logo h2 {
			font-size: 1rem;
		}

		.logo p {
			font-size: 0.75rem;
		}
	}
</style>