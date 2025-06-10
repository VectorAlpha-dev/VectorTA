<script lang="ts">
	import '../app.css';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import Footer from '$lib/components/Footer.svelte';
</script>

<div class="container">
	<Sidebar />
	
	<div class="main-content">
		<slot />
		<Footer />
	</div>
</div>